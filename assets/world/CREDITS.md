# New York terrain data

The Terrarium PNG files in `terrarium/13/` are elevation tiles from the
[Mapzen Terrain Tiles dataset hosted on AWS](https://registry.opendata.aws/terrain-tiles/).
The dataset aggregates several regional providers. For these United States
tiles, credit the U.S. Geological Survey 3D Elevation Program (3DEP), with
global GMTED2010/SRTM source credit as specified by the dataset publisher.

Required source acknowledgements:

- Mapzen / Tilezen Terrain Tiles.
- United States 3DEP and global GMTED2010/SRTM terrain data courtesy of the
  U.S. Geological Survey.

The underlying sources have their own terms. The Mapzen publisher's
[attribution and source terms](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)
are part of this data bundle and must be checked before adding tiles from
other regions. USGS 3DEP products are public domain; the USGS requests
appropriate credit.

The simulator resamples these Terrarium XYZ images onto its cube-sphere DEM
mesh. Terrarium elevations decode as `R * 256 + G + B / 256 - 32768` metres.
The bundled `terrarium/13/` mosaic contains 165 public AWS Terrain Tiles in
the New York region (XYZ columns 2404–2418, rows 3074–3084), accessed on
2026-10-31. This continuous elevation block covers the Manhattan-to-airport
terrain patch and its sampling margins. The dataset uses the same attribution
above; see the [AWS Open Data Registry](https://registry.opendata.aws/terrain-tiles/)
and [Tilezen attribution terms](https://github.com/tilezen/joerd/blob/master/docs/attribution.md).

The `osm/3/12/2635/3881.osm` New York road/building extract is © OpenStreetMap
contributors and is available under the Open Database License (ODbL). Credit
OpenStreetMap contributors; see the
[copyright and license page](https://www.openstreetmap.org/copyright).

The `osm/regions/new-york-airports.osm` regional airport-surface extract was
downloaded from OpenStreetMap through Overpass on 2026-10-31. It contains
OpenStreetMap ways tagged as runways, taxiways, aprons, helipads, aerodromes,
terminals, or hangars and their referenced nodes. It is also © OpenStreetMap
contributors and licensed under the ODbL; the same attribution and share-alike
terms apply.

The `osm/regions/new-york-roads.osm` extract contains the major OSM highway
classes (`motorway`, `trunk`, `primary`, `secondary`, and `tertiary`, including
their link ways) in the New York regional area, with referenced nodes. It was
derived from the same 2026-10-31 Overpass response and has the same ODbL
attribution and share-alike requirements.

The `osm/regions/new-york-buildings.osm` extract contains building outlines
and referenced nodes for the New York regional area. It was derived from the
same 2026-10-31 OpenStreetMap Overpass response and is © OpenStreetMap
contributors under the ODbL; the same attribution and share-alike terms apply.

The `osm/regions/new-york-water.osm` extract contains mapped water areas and
coastline ways, with their referenced nodes, for the New York region. It was
retrieved from Overpass on 2026-10-31 for the bounding box 40.40–41.02° N,
74.45–73.45° W and is © OpenStreetMap contributors under the ODbL. The
simulator draws closed water areas, multipolygon relations including inner
holes, and coastal water on the mapped water side of coastline ways.
