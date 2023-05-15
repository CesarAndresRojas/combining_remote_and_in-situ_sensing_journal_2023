# connect to the API
from datetime import date

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import Meta_Data_Writer_Sentinel_2


api = SentinelAPI('user', 'password')
footprint = geojson_to_wkt(read_geojson(
    '/Users/croja022/Library/CloudStorage/OneDrive-FloridaInternationalUniversity/Remote Sensing/src/Biscayne Bay/geoJson/BiscayneBayNorthSquare.json'))
products = api.query(footprint,
                     #date=('20220105',date(2022, 1, 5)),
                     producttype='S2MSI1C',
                     platformname='Sentinel-2',
                     # orbitdirection='ASCENDING',
                     limit=1
                     )
print(products)
# api.download_all(products)
