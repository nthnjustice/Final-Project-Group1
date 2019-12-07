def get_datainfo(source):
    if source == 'PADUS':
        return 'PADUS2_0_Shapefiles', 'https://www.sciencebase.gov/catalog/file/get/5b030c7ae4b0da30c1c1d6de?f=__disk__97%2F0a%2F32%2F970a32899eb4389aaf8b3abf61b6bc7fde229df8'
    elif source == 'MTPublicLands':
        return 'MTPublicLands_SHP', 'ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/PublicLands/MTPublicLands_SHP.zip'
    elif source == 'CONUS':
        return 'CONUS', 'https://prod-is-s3-service.s3.amazonaws.com/ScienceBase/prod/5b873238e4b0702d0e796688/812c539d11e23363d25a852f3c3cd31e7293fcc7/Species_CONUS_Range_2001v1.zip?AWSAccessKeyId=AKIAI7K4IX6D4QLARINA&Expires=1575673382&Signature=1iYpYpHDYqnMMzR7xq37VlrCzYo%3D'
    elif source == 'AMPHIBIANS':
        return 'AMPHIBIANS', 'https://drive.google.com/open?id=1b_T5ELbolYcpPep265s7Th9156fZVnSc'
    elif source == 'FW_FISH':
        return 'FW_FISH', 'https://drive.google.com/open?id=1OZAPo5QGmRVLEjUvCl3WC_u1qZ3-ZQae'
    elif source == 'MARINEFISH':
        return 'MARINEFISH', 'https://drive.google.com/open?id=1cyQ4WWVE7kSU3eeR5iUeG2pPzEUN0SWc'
    elif source == 'REPTILES':
        return 'REPTILES', 'https://drive.google.com/open?id=1Ksn5ExHY92sgpCZi8Y8CaOV_A7WqFcAG'
    elif source == 'TERRESTRIAL_MAMMALS':
        return 'TERRESTRIAL_MAMMALS', 'https://drive.google.com/open?id=13g-1xMoKLS-cCXh0AVpleciP140Sivnz',
    elif source == 'archaeology':
        return 'archaeology', 'https://github.com/SPINLab/geometry-learning/raw/develop/files/archaeology/archaeology.csv.zip'
    elif source == 'buildings':
        return 'buildings', 'https://github.com/SPINLab/geometry-learning/raw/develop/files/buildings/buildings.csv.zip'
    elif source == 'IUCN':
        return 'IUCN', 'https://drive.google.com/open?id=1qqlS5OkCdQzfFnsx-cqdjNvQdJVYIx8q'
