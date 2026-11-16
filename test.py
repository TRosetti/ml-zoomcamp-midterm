"""Send two sample sessions to the web service and print the answers.

With the service running (directly or in Docker), in another terminal:
    uv run python test.py

To test a service at another address:
    uv run python test.py http://<host>:9696/predict
"""

import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696/predict'

# a session similar to the ones that buy: visited valuable pages (high pagevalues),
# viewed many products, low exit rate, in November
likely_buyer = {
    'administrative': 4,
    'administrative_duration': 120.0,
    'informational': 1,
    'informational_duration': 30.0,
    'productrelated': 45,
    'productrelated_duration': 1800.0,
    'bouncerates': 0.0,
    'exitrates': 0.01,
    'pagevalues': 40.0,
    'specialday': 0.0,
    'month': 'Nov',
    'operatingsystems': 2,
    'browser': 2,
    'region': 1,
    'traffictype': 2,
    'visitortype': 'New_Visitor',
    'weekend': False,
}

# a session that leaves right away: a single page, maximum bounce rate, in February
likely_leaver = {
    'administrative': 0,
    'administrative_duration': 0.0,
    'informational': 0,
    'informational_duration': 0.0,
    'productrelated': 1,
    'productrelated_duration': 0.0,
    'bouncerates': 0.2,
    'exitrates': 0.2,
    'pagevalues': 0.0,
    'specialday': 0.0,
    'month': 'Feb',
    'operatingsystems': 1,
    'browser': 1,
    'region': 1,
    'traffictype': 1,
    'visitortype': 'Returning_Visitor',
    'weekend': False,
}

for name, session in [('likely buyer', likely_buyer), ('likely leaver', likely_leaver)]:
    response = requests.post(url, json=session, timeout=10)
    response.raise_for_status()
    print(f'{name}: {response.json()}')
