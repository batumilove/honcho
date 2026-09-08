"""Build the Fly deploy webhook payload and curl config without shell interpolation.

Reads GITHUB_EVENT_NAME, GITHUB_REF_NAME, INPUT_VERSION, WEBHOOK_SECRET and
WEBHOOK_URL from the environment. Writes:

- payload.json  -- JSON body posted to the webhook
- curl-config   -- curl config file carrying the URL and Authorization header,
                   so secret values never appear in argv
"""

import json
import os

event = os.environ["GITHUB_EVENT_NAME"]
if event == "workflow_dispatch":
    version = os.environ["INPUT_VERSION"]
    image_label = (
        os.environ["IMAGE_LABEL_PREFIX"] + os.environ["INPUT_VERSION"]
    )
else:
    version = os.environ["GITHUB_REF_NAME"].removeprefix("v")
    image_label = os.environ["IMAGE_LABEL_PREFIX"] + os.environ["GITHUB_REF_NAME"]

with open("payload.json", "w") as f:
    json.dump({"version": version, "image_label": image_label}, f)

# curl config quoted strings cannot contain raw double quotes; replace them.
secret = os.environ["WEBHOOK_SECRET"].replace('"', "'")
url = os.environ["WEBHOOK_URL"] + "/webhooks/v1/add_honcho_version"
config = 'header = "Content-Type: application/json"\n'
config += 'header = "Authorization: Bearer ' + secret + '"\n'
config += 'url = "' + url + '"\n'

with open("curl-config", "w") as f:
    f.write(config)
