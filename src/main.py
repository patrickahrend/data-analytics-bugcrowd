# Import the necessary libraries
from bs4 import BeautifulSoup
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options as ChromeOptions
import time


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

    # Open a CSV file for writing
    with open("programs.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Write the column headers
        writer.writerow(["Program Name", "Reward", "Program URL"])

        # Loop until pagination is set to False

        # Set the URL for the current page
        while pagination:

            driver.get(base_url + "?page=" + str(page))
            driver.implicitly_wait(10)

            # Give the page some time to load before checking for the "Next" button
            time.sleep(1)

            # Create a BeautifulSoup object from the response
            soup = BeautifulSoup(driver.page_source, "html.parser")

            program_name = soup.find_all('h4', class_='bc-panel__title')

            # Check if there are any programs on the page
            if len(program_name) > 0:
                reward_range = soup.find_all('span', class_='bc-stat__title')

                for name, range in zip(program_name, reward_range):
                    # Find the link of the program
                    program_link = name.find('a')['href']

                    writer.writerow(
                        [name.get_text(), range.get_text(), program_link])

                # Find the "Next" button
                next_button = driver.find_element(
                    By.CLASS_NAME, "bc-pagination__link")

                print("Check this out", next_button.text)
                # Get the pagination links at the bottom of the page
                pagination_links = driver.find_elements(
                    By.CLASS_NAME, "bc-pagination__number")

                # If there are no pagination links, set pagination to False
                if len(pagination_links) == 0:
                    pagination = False

                # If there are pagination links, get the page count from the last link
                else:
                    page_count = int(pagination_links[-1].text)

                    # If the current page is the last page, set pagination to False
                    if page == page_count:
                        pagination = False
                    else:
                        # If there is a "Next" button, increment the page number and click it
                        page += 1
                        print("Currently crawling Page:  "+str(page))
                        next_button.click()

            # If there are no programs on the page, set pagination to False
            else:
                pagination = False


if __name__ == "__main__":
    main()
