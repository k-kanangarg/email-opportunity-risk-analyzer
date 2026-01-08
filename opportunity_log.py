import csv
from datetime import datetime
import os

FILE_NAME = "opportunities.csv"

def log_opportunity(text, cost_risk, links):
    exists = os.path.isfile(FILE_NAME)

    with open(FILE_NAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "timestamp",
                "preview",
                "cost_risk",
                "link_count",
                "first_link"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            text[:120].replace("\n", " "),
            cost_risk,
            len(links),
            links[0] if links else ""
        ])
