# vector-space-model
an exercise of creating a vector space model

<<<<<<< HEAD
=======
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
  
  - there are 4 attribute in this object:
    1.  __documents__: based on 2048 documents collected from homedepot.com.
    2.  __tfVector__: docs embeded in a vector by term frequency formula.
    3.  __tfidfVectors__: docs embeded in a vector by tf-idf formula.
    4.  __vectorKeywordIndex__: key word that tokenize and stemmed from documents, return a dictionary contains key-value pairs(keyword:position)
    5.  __parser__: given a parser from Parser.py
  
* **main.py** 

  main exacutable function
  
* **util.py**

  utilities like tf, idf weighting, and cosine similarity and distance function

* **english.stop**

  english stop words collection
  
>>>>>>> f9a769b7ada69e8e0b1391445a8fd26efd9c2bdd
