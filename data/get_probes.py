import csv
import datetime

from ripe.atlas.cousteau import AnchorRequest

continent = 'Europe'

with open('ATLAS_API_KEY', 'r') as api_key_file:
    atlas_api_key = api_key_file.readline().rstrip()

with open('countries.csv') as countries_file:
    reader = csv.DictReader(countries_file)
    country_codes = [
        row['ISO-alpha2 Code']
        for row in reader
        if row['Region Name'] == continent
    ]

with open('probes.csv', 'w') as probes_file:
    csv_writer = csv.DictWriter(
        probes_file,
        ['id', 'city', 'country', 'latitude', 'longitude', 'fqdn', 'ipv4']
    )
    csv_writer.writeheader()
    for country_code in country_codes:
        probes = AnchorRequest(country = country_code)
        city_codes = set()
        city_names = set()
        for probe in probes:
            # Make a decent effort to skip multiple anchors in the same
            # city
            city_code = probe['fqdn'].split('-')[1]
            city = probe['city'].split(',')[0]
            if city_code in city_codes or str.lower(city) in city_names:
                continue
            city_codes.add(city_code)
            city_names.add(str.lower(city))
            csv_writer.writerow({
                'id': probe['id'],
                'city': city,
                'country': probe['country'],
                'latitude': probe['geometry']['coordinates'][1],
                'longitude': probe['geometry']['coordinates'][0],
                'fqdn': probe['fqdn'],
                'ipv4': probe['ip_v4'],
            })
