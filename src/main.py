# Import the necessary libraries
from bs4 import BeautifulSoup
import csv
import requests

# Set the URL you want to scrape
url = "https://bugcrowd.com/programs"

# Use requests to get the HTML content of the page
response = requests.get(url)

# Create a BeautifulSoup object from the response
soup = BeautifulSoup(response.text, "html.parser")

# Extract the data you want from the page using BeautifulSoup's methods
data = soup.find_all("div", {"class": "program-card__header"})

# Open a CSV file for writing
with open("programs.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write the column headers
    writer.writerow(["Program Name", "Program URL"])

    # Loop through the data and write it to the CSV file
    for item in data:
        name = item.find("h4").text
        url = item.find("a")["href"]
        writer.writerow([name, url])
