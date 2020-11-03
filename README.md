# tetris_AI
4-wide combo tetris bot.  <br/>

*spaghetti code. Needs lots of refactoring*


How it works
---
It works differently depending on the **state**.<br/>
There are two **states**, **build state** and **combo state**.<br/>
In **build state**, a neural network model called **Felix** piles tetris blocks up in 6 wide <br/>
to makes combo from it in **combo state**.
And other neural network model called **state observer** watch the game and decides when to go from one to the other.<br/>
<br/>
*will be added..*
<br/>

Performance
---
[![play video](https://share.gifyoutube.com/gZzQVj.gif)](https://www.youtube.com/watch?v=QTJNax-B11I)

* 79 line clears in minute, 12 combo in average (surpasses average human)
