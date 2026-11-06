from remotezip import RemoteZip

url = "https://zenodo.org/records/18370400/files/data.zip?download=1"

target = "data/Buildings/B01/Meters/meters.csv"

with RemoteZip(url) as z:
    matches = [name for name in z.namelist() if name.endswith("B01/Meters/meters.csv")]

    print("Found:", matches)

    if not matches:
        raise RuntimeError("B01 meters.csv not found")

    member = matches[0]

    with z.open(member) as src:
        content = src.read()

    with open("B01_meters.csv", "wb") as dst:
        dst.write(content)

print("Saved as B01_meters.csv")