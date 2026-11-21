"""Download the lab tower-crane dataset (Mendeley Data V2, CC BY 4.0) into crane/data/tower_crane_mendeley_v2/.

    python3 -I crane/data_tools/download_dataset.py

Credit: Alhassan, A., Arif, S. M. U., Talapiden, K., Shehu, M. A., Haruna, A., Do, T. (2025).
Trajectories and Payload Oscillations Data of Tower Crane Systems ... Mendeley Data, V2.
https://doi.org/10.17632/x69hyyv3zw.2  (CC BY 4.0). The data is not committed to this repo; this script fetches it.
"""
import json
import os
import urllib.request

DATASET, VERSION = 'x69hyyv3zw', 2
API = 'https://data.mendeley.com/public-api/datasets'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tower_crane_mendeley_v2')
UA = {'User-Agent': 'Mozilla/5.0 (dataset download script)'}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    folders = json.loads(get(f'{API}/{DATASET}/folders/{VERSION}'))
    files = []
    for f in [{'id': 'root'}] + folders:
        files += json.loads(get(f'{API}/{DATASET}/files?folder_id={f["id"]}&version={VERSION}'))
    for f in files:
        path = os.path.join(OUT, f['filename'].replace('/', '_'))
        if os.path.exists(path) and os.path.getsize(path) == f.get('size'):
            continue
        with open(path, 'wb') as o:
            o.write(get(f['content_details']['download_url']))
        print('got', f['filename'])
    print(f'{len(files)} files in {OUT}')


if __name__ == '__main__':
    main()
