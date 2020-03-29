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
  
* **main.py**

  main exacutable function
  
* **util.py**

  utilities like tf, idf weighting, and cosine similarity and distance function

* **english.stop**

  english stop words collection
  
