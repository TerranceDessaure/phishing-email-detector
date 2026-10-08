# phishing-email-detector
Machine learning classifier that detects phishing emails using TF-IDF and logistic regression

# Beginning the project
To start this project I opened up VS Code and opened up a new file. I opened a the folder I created that was on my desktop and ran the terminal. 

I started a project like this a couple of days ago and I was not able to get through the set up process because of some conflict with the anaconda distrution. I spent at least a couple of hours troubling shooting and I ending up just putting it aside for another day.

I began to reuse the same folder that I was using prior, so to begin I wanted to make sure the folder was empty. I did a `pwd` to show the working directory. Next I did an `ls -a` to list all the things still in the directory. It showed the `.DS_Store` which was one of the files I was working with last night.

Once I made sure the files were completely cleared, I created a New repository on Git hub. I check again to make sure there the file is still completely clear and there were some left over files from my first attempt. There are some hidden files (ex: .git, .gitignore, .venv). I just moved these hidden files to the trash and moved on to the next step in the project.

I  used this command to turn of the old environments I used in my last attempt:
<img width="472" height="117" alt="Screenshot 2026-10-04 at 6 24 28 PM" src="https://github.com/user-attachments/assets/33e30390-1caf-4d62-a2e0-3b227d207a14" />

Next I cloned my git hub repository. One thing that I learned in the process is how to ignore the dataset and the mack files using these commands

<img width="474" height="149" alt="Screenshot 2026-10-04 at 6 42 10 PM" src="https://github.com/user-attachments/assets/76da1963-b4d5-4633-90cb-f4dd436292e6" />

Then I installed the packages that I need for the project:
<img width="472" height="117" alt="Screenshot 2026-10-04 at 6 24 28 PM" src="https://github.com/user-attachments/assets/95a4ef32-7bfa-4353-a858-43301b74ec85" />

# Downloading the Dataset

I used this website to gather the csv data: https://www.kaggle.com/datasets/subhajournal/phishingemails?resource=download. Then I placed the data into its own folder. 

# Script Creation

The first thing that I did was import all the files that I would use during the project. Then I loaded the dat in to check and to make sure everything was loading correctly. 

<img width="510" height="70" alt="Screenshot 2026-10-04 at 6 56 12 PM" src="https://github.com/user-attachments/assets/24ca536c-9d30-43dc-89ac-5e78ba53e9cc" />

The data that loaded was 18,650 emails, and the emails were named either "Safe Email" or "Phishing Email". Some of the other columns were named:
- Unnamed: 0: a leftover  row-number from when the CSV was saved. It's useless, so we'll drop it.
- Email Text: the email content. Pandas hid it with `...` because it is long.
- Email Type: the label, "Safe Email" pr "Phishing Email"

# Cleaning the data

<img width="437" height="59" alt="Screenshot 2026-10-04 at 8 07 38 PM" src="https://github.com/user-attachments/assets/de47a307-8a5f-4e17-b06c-0a2e722b9878" />

First we start with `df = df.dropna()`. This will remove the rows from the data frame with missing values. Next `df["label"] = (df["label"] == "Phishing Email").astype(int)`. This is evaluates every row in the `label` column. If the text matches "Phishing Email" then it with evaluate to `True`, anything else will (like `"Safe Email"` or `"Ham"`), and it evaluates to `False`. This produces a series pf boolean values.

`.astype(int)`: this casts the boolean `True` and `False` value into integers.
- `True` becomes `1`
- `False` becomes `0`
- `df["label"] = ...`: This overwrites the original `label` column with the 1s and 0s

<img width="674" height="137" alt="Screenshot 2026-10-04 at 9 22 15 PM" src="https://github.com/user-attachments/assets/4f9a05ca-393b-470a-876b-7818601005e2" />

This function normalizes raw text so it's consistent and easier to analyze. 

## Regex
---
`import re` is the important module that involves regular expressions. A regular expression (shortened as `regex`) is a special sequence of characters that forms a search pattern used to find, check, or change text.


Some common uses for regex are:
- Data validation: Check if user input is valid email address, phone number, or postal code.
- Searching: Finding specific words, numbers, or patterns inside large blocks of text.
- Text manipulation: Quickly finding and replace specific text formats in documents  or code

## Basic Elements
- Literal character: The exact letters or numbers you want to find, such as `cat`
Metacharacters: Special symbols that represent rules or sets of character:
- `.` matches any single character.
- `\d` matches any digit from 0 to 9.
- `^` matches the start of a text string.
- `$` matches the end of a text string.

Quantifiers: Symbols that control how many times a character can repeat:
- `*` means zero or more times.
- `+` means one or more times.
- `?` means zero or one time.

`text = str(text).lower()` - converts the input to string first, so it won't crash on numbers, `None`, or missing values (common in pandas columns). Then it lowercase everything so "Free", "FREE", and "free"  are treated as the same word.

Email replacement: `\S+@\S+` means  "one or more non-space character, an @, then one or more non=space characters." Any email address becomes the word `emailtoken`. The idea is that a model usually doesn't care which email appears, only that one does.

URL replacement: `https?://\S+` matches anything starting with `http://` or `https://`  (the `s?` makes the "s" optional) up to the nest space. The `|` means "or", so `www\.\S+` also catches lines without a protocol (the \. is a literal dot). Every link becomes `urltoken`.

Number replacement: `\d+`  matches one or more digits in a row, so '42', '2026', and '5' each become `numtoken`. This stops a model form treating every distinct number as its own vocabulary word.

Punctuation removal: `[^a-z\s]` means "any character that is not a lowercase letter or whitespace. "The `^`  inside brackets negates the set. Those characters  (punctuation, symbols, emoji) are replaced with a space rather than deleted, so "hello,world" becomes "hello world" instead of "hellworld". 

Final cleanup: `\s+` matches runs of whitespace (spaces, tabs, newlines_ and collapses each run into a single space. `.strip()`  removes any leftover space at the start and end. This tidies up the gaps created by all the earlier replacements.

Why the order matters: emails and URLs are handled before punctuation removal because they rely on `@`, `:`, `/`. and `.` to be recognized. If punctuation were stripped first, those patterns  would never match. Numbers are replaced before punctuation removal too, though that order matters less.

Input like: `"Call 555-1234 or visit www.deals.comm!!"` comes out as `"call numtoken or visit urltoken"`
 




