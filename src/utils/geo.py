from __future__ import annotations
import json
from typing import List, Tuple, Optional
try:
    from shapely.geometry import shape, LineString, Polygon
except Exception:
    shape = LineString = Polygon = None

def load_first_polygon(geojson_path:str):
    with open(geojson_path,"r",encoding="utf-8") as f:
        gj = json.load(f)
    feat = gj["features"][0]
    poly = shape(feat["geometry"]) if shape else None
    return poly

def segment_crosses_land(p1, p2, poly) -> bool:
    if poly is None or LineString is None:
        return False
    line = LineString([p1, p2])
    return line.crosses(poly) or line.within(poly) or (line.touches(poly) and not line.disjoint(poly))
