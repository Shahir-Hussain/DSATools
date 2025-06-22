mod = int(1e9) + 7

class BinaryLifting:
    def __init__(self, n: int, edges: list[list[int]], root: int):
        self.n = n
        self.maxBit = n.bit_length()
        self.adj = [[] for _ in range(n+1)]
        self.depth = [0] * (n+1)
        self.table = [[0]*(n+1) for _ in range(self.maxBit)]

        for ui, vi in edges:
            self.adj[ui].append(vi)
            self.adj[vi].append(ui)
        
        self.dfs(root, 0)
        self.build()
    
    def dfs(self, node: int, parent: int) -> None:
        self.table[0][node] = parent

        for nei in self.adj[node]:
            if nei != parent:
                self.depth[nei] = self.depth[node] + 1
                self.dfs(nei, node)

        return 

    def build(self) -> None:
        for i in range(1, self.maxBit):
            for j in range(self.n + 1):
                self.table[i][j] = self.table[i-1][self.table[i-1][j]]

        return 

    def lvl(self,  node: int) -> int:
        return self.depth[node]

    def kthAncestor(self, node: int, k: int) -> int:
        for i in range(self.maxBit-1, -1, -1):
            mask = 1<<i

            if k & mask:
                node = self.table[i][node]

        return node

    def lca(self, n1: int, n2: int) -> int:
        if self.lvl(n1) > self.lvl(n2):
            n1, n2 = n2, n1

        n2 = self.kthAncestor(n2, self.lvl(n2) - self.lvl(n1))

        if n1==n2: return n1

        for i in range(self.maxBit-1, -1, -1):
            if self.table[i][n1] != self.table[i][n2]:
                n1 = self.table[i][n1]
                n2 = self.table[i][n2]

        return self.table[0][n1]


class Combinatorics:
    def __init__(self, n: int, mod: int) -> None:
        self.mod: int = mod

        self.fact = [1] * (n + 1)
        for i in range(1, n+1):
            self.fact[i] = self.fact[i-1] * i % self.mod

        self.fermet = [1] * (n + 1)
        for i in range(1, n+1):
            self.fermet[i] = pow(self.fact[i], self.mod-2, self.mod)
    
    def ncr(self, n: int, r: int):
        return self.fact[n] * self.fermet[r] * self.fermet[n-r] % self.mod

    def npr(self, n: int, r: int):
        return self.fact[n] * self.fermet[n-r] % self.mod


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent: list[int] = [i for i in range(n+1)]
        self.size: list[int] = [1 for _ in range(n+1)]
    
    def ultimateParent(self, node: int) -> int:
        if self.parent[node] != node:
            self.parent[node] = self.ultimateParent(self.parent[node])
        return self.parent[node]
    
    def unionBySize(self, node1: int, node2: int) -> None:
        n1p = self.ultimateParent(node1)
        n2p = self.ultimateParent(node2)
        if n1p == n2p: return
        if self.size[n2p] > self.size[n1p]:
            self.parent[n1p] = n2p
            self.size[n2p] += self.size[n1p]
        else:
            self.parent[n2p] = n1p
            self.size[n1p] += self.size[n2p]


class Matrix:
    def __init__(self, mod: int) -> None:
        self.mod: int = mod

    def multiplication(self, a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
        r1, c1, r2, c2 = len(a), len(a[0]), len(b), len(b[0]),
        if c1 != r2: raise Exception("Matrix Multiplication Not Possible")

        res = [[0] * c2 for _ in range(r1)]
        for i in range(r1):
            for j in range(c2):
                for k in range(c1):
                    res[i][j] = (res[i][j] + (a[i][k] * b[k][j]) % mod) % self.mod

        return res

    def exponentiation(self, m: list[list[int]], t: int) -> list[list[int]]:
        s = len(m)
        res = [[int(i==j) for j in range(s)] for i in range(s)]

        while t:
            if t&1:
                res = self.multiplication(res, m)
            m = self.multiplication(m, m)
            t >>= 1

        return res


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.end: bool = False

class Trie:
    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return curr.end

    def startsWith(self, prefix: str) -> list[str]:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]

        res = []
        def dfs(curr: TrieNode, word: str) -> None:
          if curr.end:
            res.append(word)

          for w in curr.children:
            dfs(curr.children[w], word + w)
          return 
        
        dfs(curr, prefix)
        return res


def subsets(nums: list[int]) -> list[list[int]]:
    st : set[int] = set(nums)
    dq : list[list[int]] = [[]]

    for ele in st:
        n = len(dq)

        for i in range(n):
            dq.append(dq[i] + [ele])

    return dq    


def topoSort(n: int, edges: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    adj = [[] for _ in range(n)]
    indegree = [0] * n
    res = []

    for ui, vi in edges:
        adj[ui].append(vi)
        indegree[vi] += 1

    dq = [i for i, v in enumerate(indegree) if not v]

    while dq:
        newDQ = []
        for node in dq:
            res.append(node)
            for nei in adj[node]:
                indegree[nei] -= 1
                if not indegree[nei]:
                    newDQ.append(nei)
        dq = newDQ
    
    if len(res) != n:
        res = []

    return res[::-1], adj


def pascalTriangle(n: int) -> list[list[int]]:
    comb = [[0]*n for i in range(n)]

    for i in range(n):
        comb[i][0] = 1
        for j in range(1, i + 1):
            comb[i][j] = (comb[i-1][j-1] + comb[i-1][j]) % mod

    return comb


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a%b
    return a


def SieveOfEratosthenes(n: int) -> list[bool]:
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(pow(n, 0.5)) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    
    return sieve

