import os
import pyperclip
from time import sleep
from selenium import webdriver


def translate(keyword):
    path = os.path.join(os.getcwd(), 'chromedriver')
    browser = webdriver.Chrome(executable_path=path)

    browser.get('https://www.bing.com/translator')
    sleep(2)

    german_language = browser.find_element_by_xpath('//optgroup[@id="t_srcAllLang"]/option[@value ="de"]')
    german_language.click()
    sleep(2)

    switch_language = browser.find_element_by_id('tta_revIcon')
    switch_language.click()
    sleep(2)

    search = browser.find_element_by_id('tta_input_ta')
    search.send_keys(keyword)
    sleep(2)

    copy_link = browser.find_element_by_id('tta_copyIcon')
    copy_link.click()


    with open ('text.txt','w', encoding= 'utf8') as f:
        get_text = browser.find_element_by_tag_name('textarea')
        a = get_text.text
        f.write(a)

    return get_text.text



if __name__ == '__main__':
    example = translate('Hello World')



