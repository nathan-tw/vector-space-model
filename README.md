# vector-space-model
an exercise of creating a vector space model

## usage
`$ python main.py --query {query}`

remember to put on quotation marks. e.g., 

`$ python main.py --query 'drill wood sharp'`

## files
* **Parser.py**

  clean and tokenize documents

* **PorterStemmer.py**

  the Porter stemming algorithm, ported to Python from the version coded up in ANSI C by the author.

* **ReadFiles.py**

  iterate the file and read the docs

* **VectorSpace.py**

  an object of vector space which is based on documents from input params
  
  - there are 5 attribute in this object:
    1.  __documents__: based on 2048 documents collected from homedepot.com.
    2.  __tfVector__: docs embeded in a vector by term frequency formula.
    3.  __tfidfVectors__: docs embeded in a vector by tf-idf formula.
    4.  __vectorKeywordIndex__: key word that tokenize and stemmed from documents, return a dictionary contains key-value pairs(keyword:position)
    5.  __parser__: given a parser from Parser.py
  - there is a special function **getRelevanceFeedbackVector** is used to get a feedback vector by **nltk.pos_tag**:

  
* **main.py** 

  main exacutable function

  
* **util.py**

  utilities like tf, idf weighting, and cosine similarity and distance function
  - __tf(word, wordlist)__: simply count frequency of a word in a wordlist
  - __n_containing(word, documents)__: sum all documents which contains a specific word
  - __idf(word, documents)__: log(len(documents) / (1 + n_containing(word, documents)))
  - __tfidf(word, wordlist, documents)__: tf(word, wordlist) * idf(word, documents)
  - __cosine(vector1, vector2)__: dot(vector1,vector2) / (norm(vector1) * norm(vector2)), 
  
    norm=sqrt(sum(elem**2)) for elem in vector

  - __euclidean(vector1, vector2)__: norm(vector1-vector2)
* **english.stop**

  english stop words collection

