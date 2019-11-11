import urllib
import zipfile


res = urllib.request.urlopen('ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/AdministrativeBoundaries/MontanaReservations_shp.zip').read()

with open('data/MontanaReservations.zip', 'wb') as file:
    file.write(res)

zip = zipfile.ZipFile('data/MontanaReservations.zip')
zip.extractall('data/MontanaReservations')
zip.close()