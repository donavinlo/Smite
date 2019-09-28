import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'https://smite.guru/profile/925039-Weak3n/matches?page=1'
r = requests.get(url)
# Stored in this response object
print(r.text)

# here this code parses html stored in text into a special object called soup that the beautiful soup library understands
soup = BeautifulSoup(r.text, 'html.parser')
# The previous steps was  for every web scraping project
# This helped generate the name of the gods Weak3n has played
godname = soup.find_all('div', attrs={'class': 'text-center match-widget--title'})
print(len(godname))

# To find what you want work with one of the results then apply a loop to do the rest
# If our results weren't specific already we could use the result.find() to find the specific value within the gathered
#    tag. find() searches for first matching tag and returns tag object while find_all returns result which is basically
#     a list of the data
# Can do the following if there is a tag right before the text and right after
first_god = godname[0]
# first_god.find('div class="text-center match-widget--title').text
#
# The following returns if there is no tag at all for the needed content
first_god.contents
#     This returns a python list containing its children. (Tags and strings nested within tag)
# If want to access a value within the tag acan do the following
#    first_god.find('a')['href]
#         finds the tag that starts with a and grabs the value of the attribute in brackets
# once you find how to extract each portion of the data needed can put how to extract each one in the for loop and
#      append them in a list in tuples

#Building a Dataset

