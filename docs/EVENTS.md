# AcesVision event API

AcesVision publishes what it saw. It does not know what you do about it.

This document is the normative contract between the emitter and any subscriber.
Where this document and the code disagree, that is a bug in one of them — the
tests in `test_acesvision.py` are what keep them honest.

The emitter has no knowledge of any particular subscriber. A lighting daemon, a
home-automation bridge, a logger and a notebook are the same thing to it: an
authenticated HTTP client on loopback. Nothing in this repository needs to
change to add one.

---

## 1. Transport

All endpoints are `GET`. The server binds `127.0.0.1` by default.

| Endpoint | Purpose | Auth |
|---|---|---|
| `/api/health` | Liveness. `{"ok":true,"schema":"acesvision.gesture/1"}` and nothing else. | none |
| `/api/catalog` | The gesture vocabulary and its `sha256`. | token |
| `/api/events` | Server-Sent Events. The primary subscription. | token |
| `/api/events/recent?since=<seq>` | Polling fallback. `{"events":[…],"seq":N,"oldest_available":M}`. | token |
| `/api/state` | Pipeline status for the local preview. | token |
| `/` , `/latest.jpg` | The browser preview and the current JPEG frame. | token |

`/api/health` is the only unauthenticated endpoint, and it deliberately reveals
nothing but that an emitter is running and which schema it speaks. Everything
else — including the live camera frame at `/latest.jpg` — requires the token.

### Authentication

A 32-byte URL-safe token is generated on first run into
`~/.config/acesvision/emitter.token`, mode `0600` inside a `0700` directory.
Print it with `python -m acesvision --print-token`.

Present it as either:

```
Authorization: Bearer <token>          # preferred
?token=<token>                         # for <img> and EventSource, which cannot set headers
```

The comparison is `hmac.compare_digest`. A URL-borne token ends up in shell
history and referrer headers, so use the header unless you are a browser.

If the token file is readable by group or others, the emitter refuses to use it
and tells you to delete it. It does not quietly fix the mode: a repair would
hide the fact that something already had the chance to read it.

### Refusals

| Condition | Response |
|---|---|
| Any `Origin` header present | `403` |
| `Host` is not loopback (or the explicitly bound address) | `403` |
| Missing or wrong token, on anything but `/api/health` | `401` |
| More than 8 concurrent stream subscribers | `503` + `Retry-After: 5` |
| Malformed `?since` | `400` |

`Origin` is refused outright rather than allowlisted: a browser attaches it to
cross-site requests and a legitimate subscriber has no reason to send one, so
there is no allowlist to get wrong. The `Host` check defeats DNS rebinding — a
page on an attacker's origin can make a name resolve to `127.0.0.1`, and the
socket cannot tell, but the `Host` header still carries the attacker's name. A
missing `Host` is refused; fail closed.

### Binding beyond loopback

`--bind` refuses a non-loopback address unless `--allow-remote` is also given,
and when it is, **identity publishing is forced off** — every event carries
`identity_state: "disabled"` and `actor: null`. Naming an enrolled person to
whatever is on the far end of a LAN socket is a different decision from naming
them to a process on the same machine, and a config-file default does not get to
make it.

A **wildcard** bind (`0.0.0.0`, `::`) is refused even with `--allow-remote`.
Name the interface: `--bind 192.168.1.20 --allow-remote`. The `Host` allowlist
above is built from the address that was bound, and a wildcard names none — so
the emitter would have to either answer only to `Host: 0.0.0.0`, which no client
sends and which would make the flag silently do nothing, or drop the rebinding
check. Naming the interface keeps the flag and the defence both.

---

## 2. The gesture event — `acesvision.gesture/1`

One JSON object per event.

```json
{
  "schema": "acesvision.gesture/1",
  "type": "gesture",
  "seq": 17,
  "emitted_at": 1787112196.93066,
  "emitter": {
    "id": "acesvision",
    "instance": "1a4ca9ca-dc30-4ec1-ac7e-be834b311c46",
    "version": "0.1.0",
    "host": "workstation"
  },
  "catalog": { "version": 1, "sha256": "e1243aea…" },
  "gesture": "Open_Palm",
  "confidence": 0.91,
  "held_frames": 6,
  "actor": "Toby",
  "identity_state": "identified",
  "liveness_state": "not_evaluated",
  "security_authorized": false,
  "source": {
    "id": "desk",
    "kind": "network",
    "label": "Desk cam (network: rtsp://192.168.68.40:554/h264)",
    "trusted_device": false
  },
  "frame_sequence": 117,
  "captured_at_monotonic": 29.5,
  "dropped": 0
}
```

| Field | Meaning |
|---|---|
| `schema` | Exactly `acesvision.gesture/1`. A subscriber that does not know this string must refuse the event, not guess. |
| `type` | The SSE `event:` name. `gesture`. |
| `seq` | Monotonic per emitter **instance**, starting at 1. `seq: 0` never appears, so a `seq` of 0 in a `hello` means "nothing published yet". |
| `emitted_at` | Wall clock (`time.time()`), seconds since the epoch. Comparable across machines, subject to clock skew and NTP steps. |
| `emitter.instance` | A UUID4 generated at process start. **This is how you detect a restart**: `seq` going backwards under a *new* instance is a restart; under the *same* instance it would be a bug. |
| `emitter.version` | The emitter software version. Independent of `schema`. |
| `catalog.version` / `catalog.sha256` | The vocabulary this event was produced against. See §5. |
| `gesture` | A catalog id, e.g. `Open_Palm`. Always canonical spelling. |
| `confidence` | 0..1. The two landmark-derived poses (`Middle_Finger`, `Shush`) score a flat `1.0` — they are geometric tests, not model outputs, so their confidence is not comparable to a model score. |
| `held_frames` | How many consecutive frames the pose persisted before firing. |
| `actor` | The enrolled name, **only** when `identity_state == "identified"`. `null` otherwise. |
| `identity_state` | See §3. |
| `liveness_state` | See §4. Currently always `not_evaluated`. |
| `security_authorized` | See §4. Currently always `false`. |
| `source.label` | `SourceSpec.safe_label()` — **never** the raw URL. Userinfo and query string are stripped, so an RTSP camera's embedded credentials cannot reach a subscriber. `null` when `publish_source_label` is off. |
| `source.trusted_device` | Operator-declared. A network camera is never trusted. |
| `frame_sequence` | The capture-loop frame number this gesture was recognised in. Emitter-local. |
| `captured_at_monotonic` | `time.monotonic()` **inside the emitter process**. Meaningless in any other process and not comparable to any other clock — it has no defined epoch. Use it only for deltas between two events from the same `emitter.instance`. Use `emitted_at` for anything else. |
| `dropped` | How many events *this subscriber* missed before this one. See §6. |

Fields are added to this schema only in a new version. A subscriber should
ignore fields it does not recognise, and must not assume field order.

---

## 3. `identity_state` — who gestured

Four values. This replaces an earlier collapse in which "nobody enrolled is in
frame" and "two enrolled people are in frame" both produced `actor: null`, which
made the two indistinguishable to anything downstream.

| Value | Scene | `actor` |
|---|---|---|
| `identified` | Exactly one enrolled face in frame. | the name |
| `unknown` | No enrolled face in frame. Somebody gestured; the emitter does not know who. | `null` |
| `ambiguous` | Several enrolled faces in frame. The emitter cannot say which one gestured. | `null` |
| `disabled` | The operator turned identity publishing off, or the emitter is bound beyond loopback. Says nothing about the scene. | `null` |

Internally, `events.attribute_actor` also has a `nearest` case: several enrolled
faces, resolved by which one was closest to the gesturing hand. That resolution
is used for local rule evaluation, where it is visibly marked as a guess. **It
is published as `ambiguous`**, because a proximity guess is not an
identification and a subscriber gated on `identified` must not act on one.

**Under `ambiguous`, the candidate names are not listed.** Saying "it was either
Toby or Ana" publishes both of them in order to identify neither.

No endpoint enumerates enrolled identities. There is no roster to fetch. A name
appears only attached to an event, only when the emitter is certain, and only
when the operator has left identity publishing on.

---

## 4. `liveness_state`, `security_authorized`, and the subscriber's obligation

| Field | Values | Today |
|---|---|---|
| `liveness_state` | `live` · `spoof` · `not_evaluated` | always `not_evaluated` |
| `security_authorized` | `true` · `false` | always `false` |

Nothing in this repository performs liveness detection. A printed photograph, a
phone screen, or a video call on a second monitor will produce face detections
that are indistinguishable from a person. The fields exist so that the contract
has a defined place for the answer, and so that no subscriber can read their
absence as a pass.

### Subscriber obligation — normative

A subscriber:

1. **MUST NOT** perform any action beyond *convenience* class unless
   `identity_state == "identified"`.
2. **MUST NOT** perform a *sensitive* action unless **both**
   `liveness_state == "live"` **and** `security_authorized == true`.
3. **MUST** verify `schema` before reading any other field.
4. **SHOULD** verify `catalog.sha256` against the catalog it was built against,
   and refuse or degrade on a mismatch rather than guess.

The risk classes are the ones in `acesvision/policy.py`:

* **convenience** — reversible in one gesture, harmless if triggered by the
  wrong person or by nobody. Cycling a light theme. Pausing music.
* **personal** — affects one person's environment or attention. A desktop
  switch. A notification.
* **sensitive** — anything a stranger holding up a photograph must not be able
  to do. Doors. Payments. Anything irreversible.

Because both fields are hardcoded to `not_evaluated` / `false`, **no correct
subscriber can build a sensitive binding against this emitter today.** That is
the intended state. It is written down here rather than left implicit so that a
subscriber author has to notice it, and so that turning the fields on later is a
deliberate change to a published contract rather than a quiet one.

---

## 5. The catalog

The gesture vocabulary lives in `gestures.json` and is served whole at
`/api/catalog`:

```json
{
  "catalog_version": 1,
  "gestures": [ { "id": "Open_Palm", "label": "Open palm", "builtin": true }, … ],
  "sha256": "e1243aea…"
}
```

`sha256` is taken over the canonical serialisation of the document **without**
the `sha256` key:

```python
json.dumps(document, sort_keys=True, separators=(",", ":"))
```

So it fingerprints the vocabulary's *content*, not the file's whitespace or key
order. Reformat `gestures.json` and the hash does not move; add a gesture and it
does. A subscriber can therefore reproduce the hash in any language and pin it.

Pin both. `catalog_version` is what the operator bumps and is the human-readable
handle; `sha256` is what catches the edit where the operator forgot to bump it.

---

## 6. Delivery, replay, and reconnection

### The ring

The emitter keeps the last **256** events in memory for replay. There is no
persistence: a restart loses the ring, and `emitter.instance` changes so a
subscriber can tell.

### SSE framing

```
id: 17
event: gesture
data: {"schema":"acesvision.gesture/1",…}

: keepalive

```

A `: keepalive` comment is sent after **15 seconds** of silence. Without it an
idle emitter and a dead one are identical to a subscriber, and the difference
matters: one means "nobody has gestured", the other means "you will not hear
about it when they do". Clients ignore comment lines; treat a gap longer than
~2× the keepalive interval as a dead connection and reconnect.

### `hello`

Every connection opens with one `hello` frame, so a subscriber bootstraps in a
single round trip:

```json
{
  "schema": "acesvision.hello/1",
  "type": "hello",
  "emitted_at": 1787112193.9,
  "emitter": { "id": "acesvision", "instance": "…", "version": "0.1.0", "host": "…" },
  "catalog": { "catalog_version": 1, "gestures": [ … ], "sha256": "…" },
  "seq": 14,
  "supported_schemas": ["acesvision.gesture/1"],
  "publishing": true,
  "publish_filter": { "enabled": true, "gestures": null, "min_confidence": 0.0,
                      "publish_identity": true, "publish_source_label": true },
  "replay": { "requested_since": 20, "oldest_available": 1, "gap": false }
}
```

`publishing: false` means the emitter was started with `--no-emit`. The stream
is healthy and will stay silent. This is why `--no-emit` still serves the
endpoint rather than closing it: "switched off" must read differently from
"dead".

`replay` appears only when replay was requested. `gap: true` means the ring no
longer holds the point you asked to resume from, so events were lost before you
connected. Handle it — the alternative is being handed a short history that
looks complete.

The `hello` frame's `id:` is the sequence you are caught up to, so a connection
that drops before the first real event still reconnects from the right place.

### Resuming

| Mechanism | Precedence |
|---|---|
| `Last-Event-ID: <seq>` header | wins |
| `?since=<seq>` query | used only when the header is absent |

The header wins because a browser `EventSource` re-requests the *original URL*
on reconnect and adds the header. Honouring the query string would replay from
the connection's original point every time and duplicate everything since.

A malformed `?since` is a `400` — you typed it. A malformed `Last-Event-ID` is
ignored — your HTTP client sent it.

Replay is `seq > since`. Registering the subscription and taking the replay
slice happen under one lock, so there is no window in which an event is both
replayed and delivered live, or neither.

### Back-pressure: `dropped`

The emitter never blocks on a subscriber. Each subscriber has a **64**-event
queue; when it is full the **oldest** queued event is discarded and `dropped` is
incremented. Newest-wins is deliberate: a gesture from four seconds ago is worth
less than the one that just happened, and a subscriber that has fallen behind is
better off current than complete.

`dropped` is stamped per subscriber at delivery, so the same `seq` handed to two
subscribers may carry two different counts — it describes *your* gap, not the
emitter's. On the replay and polling paths it is always `0`; there, a gap is
visible directly as a gap in `seq`.

This mirrors the capture loop one level down: `pipeline._OutputWorker` gives each
output a one-slot mailbox that overwrites rather than waits, so a slow consumer
costs frames, never latency.

### Polling fallback

```
GET /api/events/recent?since=17
→ {"events": [ … ], "seq": 27, "oldest_available": 1}
```

For a subscriber that cannot hold a connection open. Omitting `since` returns no
events and just tells you the current `seq` to resume from — a first poll gets a
starting point, not a ring dump it has no context for. Compare `oldest_available`
against your `since` to detect a gap.

---

## 7. Operator control — `~/.config/acesvision/publish.json`

```json
{
  "enabled": true,
  "gestures": ["Open_Palm", "Closed_Fist"],
  "min_confidence": 0.6,
  "publish_identity": true,
  "publish_source_label": true
}
```

| Key | Default | Effect |
|---|---|---|
| `enabled` | `true` | `false` publishes nothing at all. |
| `gestures` | `null` | `null`/absent = the whole vocabulary. A list = **exactly** those. An empty list therefore means *none*, not *all*. |
| `min_confidence` | `0.0` | Events below this are not published. |
| `publish_identity` | `true` | `false` forces `identity_state: "disabled"` and `actor: null` on every event. |
| `publish_source_label` | `true` | `false` sets `source.label` to `null`; `id` and `kind` remain, and neither carries a hostname. |

The file is optional; absent means the defaults, which are the behaviour that
already existed. A present file can only narrow.

Allowlist entries are matched against the catalog ignoring case and separators,
so `open_palm` resolves to `Open_Palm`. An entry that is not in the catalog at
all is dropped with a warning at startup rather than raising: an unknown name in
an allowlist can only ever fail closed, and one typo should not take the emitter
down.

---

## 8. A minimal subscriber

```bash
python -m acesvision --print-token
```

```python
import json, requests

TOKEN = "…"
EXPECTED_CATALOG_SHA = "e1243aea…"

with requests.get("http://127.0.0.1:8765/api/events",
                  headers={"Authorization": f"Bearer {TOKEN}"},
                  stream=True, timeout=(5, 40)) as response:
    response.raise_for_status()
    name, data = None, None
    for line in response.iter_lines(decode_unicode=True):
        if line is None or line.startswith(":"):
            continue                       # keepalive
        if line == "":                     # frame boundary
            if name == "hello":
                assert data["catalog"]["sha256"] == EXPECTED_CATALOG_SHA
            elif name == "gesture":
                assert data["schema"] == "acesvision.gesture/1"
                if data["identity_state"] == "identified":
                    handle(data["gesture"], data["actor"])
                else:
                    handle_convenience_only(data["gesture"])
            name, data = None, None
            continue
        field, _, value = line.partition(": ")
        if field == "event":
            name = value
        elif field == "data":
            data = json.loads(value)
```

Track the last `id:` you processed and send it back as `Last-Event-ID` when you
reconnect.
