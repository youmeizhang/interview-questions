### 1. ATM
(credit to: https://zhuanlan.zhihu.com/p/33736278)

#### Clarify
* Input: card, Ouput: cash
* Limitation of Input: only debit card
* Limitation of Output: must be in multiples of $20
* Float balance sufficient: yes, it has enough balance
* Input: might be in multiple accounts, so you can select account

#### Core objects
* Debit Card
* ATM Machine
* Account

#### Use cases
* Get debit card
* Authorization
* Select account
* Check balance
* Deposite money
* Withdraw money
* Log out

#### Classes
##### ATM Machine
* float balance
* session
  - takeDebitCard
  - login
  - selectAccount
  - checkBalance
  - depositeMoney
  - withdrawMoney
  - logout
  ```Python
  class machine():
    def __init__(self):
        self.dic = {}
        self.account = {}
        self.accountId = 0
        
    def addAccount(self, id, password, accountId):
        if id in self.dic:
            print("This card is already registered")
            self.account[id].append(accountId)
        else:
            self.dic[id] = MD5(password)
            self.account[id] = [accountId]
    
    def login(id, password):
        if self.dic[id] == MD5(password):
            return True
        else:
            return False
    
    def selectAccount(id, accountId):
        if accountId in self.account[id]:
            self.accountId = accountId
        else:
            print("This account does not exist")
  ```
##### Session
* DebitCard currentDebitCard
* Account currentAccount

##### Account
* float balance
  - depositeMoney
  - withdrawMoney
  ```Python
  class account():
    def __init__(self, id, balance = 0):
        self.id = id
        self.balance = balance
        
    def getBalance(self):
        return self.balance
    
    def depositMoney(self, amount):
        self.balance += amount
        
    def withdrawMoney(self, amount):
        if amount < self.balance:
            print("insufficient balance")
        elif amount % 20 != 0:
            print("amount should be in multiples of 20")
        else:
            self.balance -= amount
  ```
#### Pattern
  * State Design Pattern
