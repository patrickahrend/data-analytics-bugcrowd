# Import the necessary libraries
from bs4 import BeautifulSoup
import csv
import requests

# Set the URL you want to scrape
base_url = "https://bugcrowd.com/programs"

# Set the page number
page = 1

# Set the pagination flag to True
pagination = True

# Open a CSV file for writing
with open("programs.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write the column headers
    writer.writerow(["Program Name", "Reward", "Program URL"])

    # Loop until pagination is set to False
    while pagination:
        # Set the URL for the current page
        url = base_url + "?page=" + str(page)

        # Use requests to get the HTML content of the page
        response = requests.get(url)

        # Create a BeautifulSoup object from the response
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the data you want from the page using BeautifulSoup's methods
        data = soup.find_all("div", {"class": "program-card__header"})

        # Loop through the data and write it to the CSV file
        for item in data:
            name = item.find("h4").text
            url = item.find("a")["href"]
            reward = item.find("p", {"class": "program-card__reward"}).text
            writer.writerow([name, reward, url])

        # Check if there is a "Next" button on the page
        next_button = soup.find(
            "a", {"class": "pagination__link", "rel": "next"})

        # If there is no "Next" button, set pagination to False
        if next_button is None:
            pagination = False
        else:
            # If there is a "Next" button, increment the page number
            page += 1
