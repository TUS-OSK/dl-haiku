import sys
import tkinter as tk

root = tk.Tk()
root.title(u"select kigo")
width = 800
height = 600
root.geometry("{0}x{1}".format(width,height))

kigo = "桜"

def SelectKigo(x):
    static = tk.Label(text=kigo)
    static.pack(side=tk.BOTTOM)

Button = tk.Button(text=kigo, width=10)
Button.bind("<Button-1>",SelectKigo)

Button.pack()

root.mainloop()