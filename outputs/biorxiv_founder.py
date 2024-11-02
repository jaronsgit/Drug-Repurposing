import json
from urllib import request
from bs4 import BeautifulSoup
from tabulate import tabulate

DEFAULT_URL = {
    'biorxiv':
    'https://www.biorxiv.org/search/{}%20numresults%3A10%20sort%3Ascore%20direction%3Adescending'
}


class BiorxivFounder():
    def __init__(self):
        self.search_engine = 'biorxiv'
        self.serach_url = DEFAULT_URL[self.search_engine]
        return

    def _get_article_content(self,
                             page_soup,
                             exclude=[
                                 'abstract', 'ack', 'fn-group', 'ref-list'
                             ]):
        article = page_soup.find("div", {'class': 'article'})
        article_txt = ""
        if article is not None:
            for section in article.children:
                if section.has_attr('class') and any(
                        [ex in section.get('class') for ex in exclude]):
                    continue
                article_txt += section.get_text(' ')

        return article_txt

    def _get_all_links(self, page_soup, base_url="https://www.biorxiv.org"):
        links = []
        for link in page_soup.find_all(
                "a", {"class": "highwire-cite-linked-title"}):
            uri = link.get('href')
            links.append({'title': link.text, 'biorxiv_url': base_url + uri})

        return links

    def _get_papers_list_biorxiv(self, query):
        papers = []
        url = self.serach_url.format(query)
        page_html = request.urlopen(url).read().decode("utf-8")
        page_soup = BeautifulSoup(page_html, "lxml")
        links = self._get_all_links(page_soup)
        papers.extend(links[:10])  # takes only 10 most relevant papers
        return papers

    def query(self, query, metadata=False, full_text=False):
        query = query.replace(' ', '%20')

        papers = self._get_papers_list_biorxiv(query)

        return papers
    

    def display_results(self, results):
        # Prepare data for tabulate
        table_data = [[i+1, paper['title'][:100] + "..." if len(paper['title']) > 100 else paper['title'], 
                      paper['biorxiv_url']] 
                     for i, paper in enumerate(results)]
        
        # Create and print table
        headers = ["#", "Title", "URL"]
        print("\nBiorXiv Search Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    

retriever = BiorxivFounder()

# Query keywords
query = "remdesivir AND (mechanism of action OR ADME OR ebola)"

results = retriever.query(query=query)

retriever.display_results(results)
