Baby A3C: solving Atari environments in 180 lines
=======
Sam Greydanus | October 2017 | MIT License

Results after training on 40M frames:

![breakout-v0.gif](breakout-v0/breakout-v0.gif)
![pong-v0.gif](pong-v0/pong-v0.gif)
![spaceinvaders-v0.gif](spaceinvaders-v0/spaceinvaders-v0.gif)

Usage
--------

If you're working on OpenAI's [Breakout-v0](https://gym.openai.com/envs/Breakout-v0/) environment:
 * To train: `python baby-a3c.py --env Breakout-v0`
 * To test: `python baby-a3c.py --env Breakout-v0 --test True`
 * To render: `python baby-a3c.py --env Breakout-v0 --render True`

About
--------

_Make things as simple as possible, but not simpler._

Frustrated by the number of deep RL implementations that are clunky and opaque? In this repo, I've stripped a [high-performance A3C model](https://github.com/ikostrikov/pytorch-a3c) down to its bare essentials. Everything you'll need is contained in 180 lines...
	
 * If you are trying to **learn deep RL**, the code is compact, readable, and commented
 * If you want **quick results**, I've included pretrained models
 * If **something goes wrong**, there's not a mountain of code to debug
 * If you want to **try something new**, this is a simple and strong baseline

|			                         | Breakout-v0  | Pong-v0       | SpaceInvaders-v0  |
| -------------                      |:------------:| :------------:| :------------:    |
| *Mean episode rewards @ 40M frames | 290 ± 10     | 20.2 ± 0.1    |   425 ± 10        |
| *Mean episode rewards @ 80M frames | 320 ± 10     | 20.2 ± 0.1    |   425 ± 10        |

\*same (default) hyperparameters across all environments

Environments that work
--------
_(Use `pip freeze` to check your environment settings)_
 * Mac OSX or Linux
 * Python 2.7 or 3.6
 * NumPy 1.13.1
 * Gym 0.9.4
 * SciPy 0.19.1 (just on two lines -> workarounds possible)
 * [PyTorch 0.2.0](http://pytorch.org/)

Known issues
--------
 * Python 2.7 + PyTorch 0.2.0 + Mac OSX produces a **segfault**
   * workaround: revert to PyTorch 0.1.12 (`pip2 install http://download.pytorch.org/whl/torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl`)