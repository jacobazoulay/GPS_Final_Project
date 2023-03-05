import requests
import sys

url = sys.argv[1]

#Adds '*?list' to the end of URL if not included already
if not url.endswith("*?list"):
    url = url + "*?list"

#Makes request of URL, stores response in variable r
r = requests.get(url)

#Prints the results of the directory listing
print(r.text)