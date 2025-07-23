class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        return f"{self.title} by {self.author}"

    def __repr__(self):
        return f"<{self.title} at {id(self)}>"

    def __len__(self):
        return self.pages

    def __eq__(self, other):
        return (
            self.title == other.title and
            self.author == other.author
        )

# 使用 Book 类
book1 = Book("《Python之光》", "李老师", 500)
book2 = Book("《Python之光》", "李老师", 500)
book3 = Book("《深入浅出Pandas》", "李老师", 600)

print(book1) # 《Python之光》 by 李老师
print(len(book1)) # 500

print(book1 == book2) # True
print(book2 == book3) #  False

print(repr(book3)) # '<《深入浅出Pandas》 at 4409718064>'