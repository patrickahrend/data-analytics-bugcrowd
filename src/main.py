# Import the necessary libraries
from bs4 import BeautifulSoup
import csv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options as ChromeOptions

from selenium.webdriver.chrome.service import Service


def main():
    # Set the URL you want to scrape
    base_url = "https://bugcrowd.com/programs"

    # Set the page number
    page = 1

    # Set the pagination flag to True
    pagination = True

    s = Service('/usr/local/bin/chromedriver')

    chromeOptions = ChromeOptions()
    chromeOptions.headless = True

    driver = webdriver.Chrome(service=s, options=chromeOptions)
    driver.implicitly_wait(10)

    # with open("programs.html", "w") as file:
    #     driver.get("https://bugcrowd.com/programs")
    #     driver.implicitly_wait(10)

    #     # Create a BeautifulSoup object from the response
    #     soup = BeautifulSoup(driver.page_source, "html.parser")
    #     file.write(soup.prettify())
    #     program_name = soup.find_all('h4', class_='bc-panel__title')
    #     reward_range = soup.find_all('span', class_='bc-stat__title')#

    #     for name, range in zip(program_name, reward_range):
    #         print(name.get_text(), range.get_text())

    # Open a CSV file for writing
    with open("programs.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Write the column headers
        writer.writerow(["Program Name", "Reward", "Program URL"])

        # Loop until pagination is set to False
        while pagination:
            # Set the URL for the current page
            url = base_url + "?page=" + str(page)

            driver.get("https://bugcrowd.com/programs")
            driver.implicitly_wait(10)

            # Create a BeautifulSoup object from the response
            soup = BeautifulSoup(driver.page_source, "html.parser")

            file.write(soup.prettify())
            program_name = soup.find_all('h4', class_='bc-panel__title')
            reward_range = soup.find_all('span', class_='bc-stat__title')

            with open("programs.csv", "w", newline="") as file:
                writer = csv.writer(file)

            # Write the column headers
                writer.writerow(
                    ["Program Name", "Reward Range", "Program URL"])

                for name, range in zip(program_name, reward_range):
                    # Find the link of the program
                    program_link = name.find('a')['href']

                    writer.writerow(
                        [name.get_text(), range.get_text(), program_link])

            # Check if there is a "Next" button on the page
            next_button = soup.find(
                "a", {"class": "pagination__link", "rel": "next"})

            # If there is no "Next" button, set pagination to False
            if next_button is None:
                pagination = False
            else:
                # If there is a "Next" button, increment the page number
                page += 1


if __name__ == "__main__":
    main()
