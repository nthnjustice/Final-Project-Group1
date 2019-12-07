def get_datainfo(source):
    if source == 'buildings':
        return 'buildings', 'https://github.com/SPINLab/geometry-learning/raw/develop/files/buildings/buildings.csv.zip'
    elif source == 'archaeology':
        return 'archaeology', 'https://github.com/SPINLab/geometry-learning/raw/develop/files/archaeology/archaeology.csv.zip'
    elif source == 'PADUS':
        return 'PADUS2_0_Shapefiles', 'https://www.sciencebase.gov/catalog/file/get/5b030c7ae4b0da30c1c1d6de?f=__disk__97%2F0a%2F32%2F970a32899eb4389aaf8b3abf61b6bc7fde229df8'
    elif source == 'MTPublicLands':
        return 'MTPublicLands_SHP', 'ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/PublicLands/MTPublicLands_SHP.zip'
    elif source == 'CONUS':
        return 'Species_CONUS_Range_2001v1', 'https://gwu-ml2.s3.amazonaws.com/final-project-data/CONUS.zip'
    elif source == 'IUCN':
        return 'IUCN', 'https://gwu-ml2.s3.amazonaws.com/final-project-data/IUCN.zip'
    else:
        raise Exception("{} is an invalid source option".format(source))
