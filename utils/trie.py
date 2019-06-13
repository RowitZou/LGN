import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)   #键值不存在时返回TrieNode()
        self.is_word = False

class Trie:    #大概是用来存词的一个类，insert存入相连的字组成的词，search可以查找一串字是否组成一个已存的词
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]   #如果current的children中不存在letter，则在children中插入[letter]=TrideNode()
                                                 #root的children存的是所有为词开头的字，每次一开始给root的children插入了首字
        current.is_word = True   #此时current指向最后一个字，将其is_word=true

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)    #get,如果键值不存在返回None。即如果当前词到current后面一个字不在词典中，返回None
            if current is None:
                return False
        return current.is_word     #如果到最后一个字，并不是某个词的结尾，也会返回None

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’
        matched = []
        ## while len(word) > 1 does not keep character itself, while word keed character itself
        while len(word) > 1:
            if self.search(word):   #如果当前序列组成的词在词典中，就连接成词加入matched
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

