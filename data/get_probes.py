import csv

from ripe.atlas.cousteau import AnchorRequest, ProbeRequest, MeasurementRequest

continent = 'Europe'

with open('countries.csv', 'r') as countries_file:
    reader = csv.DictReader(countries_file)
    countries = {
        row['ISO-alpha2 Code']: row['Country or Area']
        for row in reader
        if row['Region Name'] == continent
    }

with open('probes.csv', 'w') as probes_file:
    csv_writer = csv.DictWriter(
        probes_file,
        [
            'id', 'city', 'country', 'latitude', 'longitude',
            'fqdn', 'ipv4', 'measurement_id'
        ]
    )
    csv_writer.writeheader()
    for (country_code, country_name) in countries.items():
        print(country_name)
        anchors = {
            anchor['probe']: anchor
            for anchor in AnchorRequest(country = country_code)
            if not anchor['is_disabled']
        }
        probes = ProbeRequest(
            country_code = country_code,
            is_public = True,
            is_anchor = True,
            status = 1
        )
        city_codes = set()
        city_names = set()
        for probe in probes:
            # Probes should be associated to an anchor by our filtering
            # (though, this sometimes fails for some reason)
            probe_id = probe['id']
            if probe_id not in anchors:
                continue
            anchor = anchors[probe_id]

            # Get the measurement ID for pings targeting this probe.
            # Some of the probes don't have associated measurements, so
            # we skip them.
            address = probe['address_v4']
            measurement_id = None
            for measurement in MeasurementRequest(
                target_ip = address,
                is_public = True,
                type = 'ping',
                tags = ['anchoring', 'mesh'],
                status = 2
            ):
                if measurement['description'].startswith(
                    'Anchoring Mesh Measurement'
                ):
                    measurement_id = measurement['id']
                    break
            if measurement_id is None:
                continue

            # Make a decent effort to skip multiple anchors in the same
            # city
            city_code = anchor['fqdn'].split('-')[1]
            city = anchor['city'].split(',')[0]
            should_skip = False
            if city_code in city_codes or str.lower(city) in city_names:
                should_skip = True
            city_codes.add(city_code)
            city_names.add(str.lower(city))
            if should_skip:
                continue

            csv_writer.writerow({
                'id': probe_id,
                'city': city,
                'country': country_name,
                'latitude': probe['geometry']['coordinates'][1],
                'longitude': probe['geometry']['coordinates'][0],
                'fqdn': anchor['fqdn'],
                'ipv4': address,
                'measurement_id': measurement_id,
            })
