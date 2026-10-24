"""The stored chain's shape — the contract between the collector and the page.

ThetaData returns ``right`` as ``CALL``/``PUT`` and dates as ``datetime.date``;
the collector normalises to this layout once, at ingest, so nothing downstream
branches on the vendor's spelling.
"""

from __future__ import annotations

SYMBOL = "symbol"
SESSION = "session"
EXPIRATION = "expiration"
STRIKE = "strike"
RIGHT = "right"
BID = "bid"
ASK = "ask"
CLOSE = "close"
VOLUME = "volume"
BID_SIZE = "bid_size"
ASK_SIZE = "ask_size"

#: Every column a stored chain carries, in write order.
COLUMNS = (SYMBOL, SESSION, EXPIRATION, STRIKE, RIGHT, BID, ASK, CLOSE, VOLUME, BID_SIZE, ASK_SIZE)

CALL = "C"
PUT = "P"
