#!/usr/bin/env python3
"""
toggle_daemon.py

Receptor HTTP que acepta POST /event con JSON:
{
  "msg_id": "...",
  "source": "detector",
  "zone": "A1",
  "event": "zone_hold",
  "hold_seconds": 3.0,
  "p_presence": 0.86,
  "p_pointing": 0.72,
  "ts": 169xxx.xxxxx
}

Al recibirlo, hace toggle en Home Assistant:
 - consulta GET /api/states/<entity_id>
 - si "off" -> POST services/<domain>/turn_on
 - if "on"  -> POST services/<domain>/turn_off

Config via ENV:
 - HA_TOKEN (required)  -> eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjYzZkY2NlMjc3ODQ0MTg3YTExOTgzYzBlYjE1ZmZhYSIsImlhdCI6MTc1Njc0Njk5NiwiZXhwIjoyMDcyMTA2OTk2fQ.j1jzTDoM4qwCACk3UZxPTeEkLA8AgS_wTliFfdbXb4Q
 - HA_URL  (optional)   -> default: http://192.168.100.223:8123
 - BIND_HOST (optional) -> default: 127.0.0.1
 - BIND_PORT (optional) -> default: 5000
"""

import os
import time
import json
import threading
import logging
from collections import OrderedDict

import requests
from flask import Flask, request, jsonify

# --- Config (env override)
HA_URL = os.getenv("HA_URL", "http://192.168.31.153:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjYzZkY2NlMjc3ODQ0MTg3YTExOTgzYzBlYjE1ZmZhYSIsImlhdCI6MTc1Njc0Njk5NiwiZXhwIjoyMDcyMTA2OTk2fQ.j1jzTDoM4qwCACk3UZxPTeEkLA8AgS_wTliFfdbXb4Q")   # MUST be set by the user
BIND_HOST = os.getenv("BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.getenv("BIND_PORT", "5000"))

# Behavior parameters
DEDUP_CACHE_SIZE = 2000        # keep last N msg_ids
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_BASE = 0.5      # seconds: 0.5,1,2,4
REQUEST_TIMEOUT = 3.0         # seconds for HA requests

# Map zone -> entity_id in HA (edit if needed)
ZONE_TO_ENTITY = {
    "C1": "media_player.chromecast_tv_movil",
    "A3": "media_player.epson_4_14",   # you gave this entity
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("toggle_daemon")

if HA_TOKEN is None:
    log.error("Environment variable HA_TOKEN is not set. Exiting.")
    raise SystemExit(1)

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

app = Flask(__name__)

# dedupe LRU: keep recent msg ids
class LRUCache:
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self._od = OrderedDict()
        self._lock = threading.Lock()

    def add(self, key):
        with self._lock:
            if key in self._od:
                self._od.move_to_end(key)
                return False
            self._od[key] = True
            if len(self._od) > self.maxsize:
                self._od.popitem(last=False)
            return True

    def contains(self, key):
        with self._lock:
            return key in self._od

recent_msgs = LRUCache(maxsize=DEDUP_CACHE_SIZE)

# per-zone state and locks
zone_state = {}  # zone -> {'is_on': bool, 'last_change_ts': float}
zone_locks = {}  # zone -> threading.Lock()

def ensure_zone(z):
    if z not in zone_state:
        zone_state[z] = {"is_on": None, "last_change_ts": None}
    if z not in zone_locks:
        zone_locks[z] = threading.Lock()

# --- HA helper functions
def ha_get_entity_state(entity_id):
    """Return 'on' or 'off' or None on error."""
    url = f"{HA_URL}/api/states/{entity_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            return j.get("state")
        else:
            log.warning("GET state %s -> status %d", entity_id, r.status_code)
            return None
    except Exception as e:
        log.exception("Error GET state %s: %s", entity_id, e)
        return None

def ha_call_service(domain, service, payload):
    """Call HA service with retries. Returns True on success."""
    svc_url = f"{HA_URL}/api/services/{domain}/{service}"
    backoff = RETRY_BACKOFF_BASE
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = requests.post(svc_url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
            if 200 <= r.status_code < 300:
                return True
            else:
                log.warning("HA service %s/%s returned %d. attempt %d", domain, service, r.status_code, attempt)
        except Exception as e:
            log.warning("HA call exception %s/%s attempt %d: %s", domain, service, attempt, e)
        time.sleep(backoff)
        backoff *= 2
    return False

def toggle_entity(entity_id):
    """
    Query current state and toggle (on -> off, off -> on).
    Returns True on success, False on failure or unknown state.
    """
    domain = entity_id.split(".")[0]
    # determine on/off services: domain/turn_on and domain/turn_off normally exist
    # e.g. input_boolean.turn_on, media_player.turn_on, light.turn_on, switch.turn_on
    current = ha_get_entity_state(entity_id)
    if current is None:
        log.error("Could not get state for %s", entity_id)
        return False

    target = None
    if current.lower() in ("on", "playing", "paused"):
        # For media_player 'playing'/'paused' we treat as ON
        service = "turn_off"
    else:
        service = "turn_on"

    payload = {"entity_id": entity_id}
    ok = ha_call_service(domain, service, payload)
    if ok:
        log.info("Toggled %s -> %s via %s/%s", entity_id, "on" if service=="turn_on" else "off", domain, service)
        return True
    else:
        log.error("Failed toggling %s via %s/%s", entity_id, domain, service)
        return False

# --- event processing
def process_zone_hold_message(msg):
    """
    msg: dict parsed from JSON with fields 'msg_id','zone','p_presence','p_pointing','ts'
    """
    msg_id = msg.get("msg_id")
    zone = msg.get("zone")
    ts = msg.get("ts", time.time())

    if not msg_id or not zone:
        log.warning("Bad message format, missing msg_id or zone: %s", msg)
        return {"status":"bad_request"}, 400

    # dedupe
    if not recent_msgs.add(msg_id):
        log.info("Duplicate msg_id %s ignored", msg_id)
        return {"status":"dup"}, 200

    # ensure zone mapping
    if zone not in ZONE_TO_ENTITY:
        log.warning("Zone %s not mapped to entity. Ignoring.", zone)
        return {"status":"unknown_zone"}, 200

    entity = ZONE_TO_ENTITY[zone]

    ensure_zone(zone)
    lock = zone_locks[zone]

    # lock per zone to avoid racing toggles
    with lock:
        st = zone_state[zone]
        # if is_on is None -> fetch current from HA to initialize
        if st["is_on"] is None:
            state = ha_get_entity_state(entity)
            if state is None:
                # couldn't query HA; but don't drop the message - try toggle anyway with retries
                log.warning("Could not query initial state for %s. Proceeding to toggle attempt.", entity)
            else:
                st["is_on"] = (state.lower() == "on")
                log.info("Initialized state for %s -> %s", zone, st["is_on"])

        # toggle: if currently on -> turn off, else turn on
        target_on = not bool(st["is_on"])
        # call HA
        ok = toggle_entity(entity)
        if ok:
            st["is_on"] = target_on
            st["last_change_ts"] = time.time()
            return {"status":"toggled", "zone":zone, "state": "on" if st["is_on"] else "off"}, 200
        else:
            return {"status":"ha_call_failed"}, 500

# --- Flask endpoints
@app.route("/event", methods=["POST"])
def event():
    try:
        msg = request.get_json(force=True)
    except Exception as e:
        log.warning("Bad JSON: %s", e)
        return jsonify({"status":"bad_json"}), 400

    # optional: simple validation
    required = ("msg_id", "zone", "event")
    if not all(k in msg for k in required):
        return jsonify({"status":"missing_fields"}), 400

    if msg.get("event") != "zone_hold":
        log.info("Ignoring non-zone_hold event: %s", msg.get("event"))
        return jsonify({"status":"ignored_event"}), 200

    resp, code = process_zone_hold_message(msg)
    return jsonify(resp), code

@app.route("/status", methods=["GET"])
def status():
    # return zone_state summary
    s = {z: {"is_on": zone_state[z]["is_on"], "last_change_ts": zone_state[z]["last_change_ts"]} for z in zone_state}
    return jsonify({"zones": s, "recent_msgs_count": len(recent_msgs._od)}), 200

if __name__ == "__main__":
    log.info("Starting toggle_daemon on %s:%d", BIND_HOST, BIND_PORT)
    app.run(host=BIND_HOST, port=BIND_PORT)
