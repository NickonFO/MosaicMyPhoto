from gooey import *
import argparse
from icrawler.builtin import GoogleImageCrawler
# Program using icrawler to gather your own database of tiles
# from google's search engine.
@Gooey
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--database-directory",dest='database',type = str,required=True,help = "directory for which image database will be stored")
        parser.add_argument("--search-term",dest="search_term",type = str,required=True,help="Enter search term for the kind of tiles you want to gather")

        args=parser.parse_args()
        google_crawler = GoogleImageCrawler(storage={'root_dir': args.database})
        google_crawler.crawl(keyword=args.search_term, max_num=100)

main()
