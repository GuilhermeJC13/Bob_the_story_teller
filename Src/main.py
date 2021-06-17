from chatbot import ChatBot
from interface import Interface
import threading

window = Interface()
bot = ChatBot()

flag = True

render_thread = threading.Thread(target=bot.listen, args=(window,))
render_thread.start()

while flag:
    window.render()