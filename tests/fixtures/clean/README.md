# Cleaner fixtures

Pages for `tests/test_clean.py`, which lists the text that must survive cleaning and the
text that must go for each.

## Captured pages

Real pages from the practice sites the owner approved for committing (#98): Zyte's
public web-scraping sandboxes. Neither site has a `robots.txt`. Saved on 2026-09-30 with
`curl`, unmodified.

| File | Source |
| --- | --- |
| `books_home.html` | https://books.toscrape.com/index.html |
| `books_category_travel.html` | https://books.toscrape.com/catalogue/category/books/travel_2/index.html |
| `quotes_home.html` | https://quotes.toscrape.com/ |
| `quotes_author_einstein.html` | https://quotes.toscrape.com/author/Albert-Einstein/ |
| `quotes_tag_love.html` | https://quotes.toscrape.com/tag/love/ |
| `quotes_js.html` | https://quotes.toscrape.com/js/ |

Product pages from books.toscrape.com are in `tests/fixtures/books/`. The synthetic test
site (`jevex.testsite`) isn't captured: the tests render it from a fixed seed, so its
pages can't go stale.

## Hand-built pages

Modelled on common templates, because real commercial pages aren't committed:
`wordpress_post.html` (WordPress with Cookiebot), `shop_product.html` (Shopify with
OneTrust), `news_article.html` (a windows-1252 news page with Sourcepoint and Quantcast)
and `books_catalogue.html` (a trimmed books.toscrape.com product page).
