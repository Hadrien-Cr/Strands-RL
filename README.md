In this repo, I experimented with Reinforcement learning on the game of Strands.
Strands is a simple strategy game where players take turns covering empty spaces on a hexagonal grid. It has similarities to Go.

## Rules
One player has white stones, the other player has black. The game is played on a hexagonal grid, with hexagons marked with the numbers from 1 to 6

1. Start by covering any spaces marked "2" with Black.

2. Then, starting with White, take turns covering up to X empty spaces marked "X". For example, you could cover any 3 empty spaces marked "3".

3. If the board is full, the game ends. The player with the largest contiguous group of stones wins. If tied, compare the players' second-largest groups, and so on, until you come to a pair which aren't the same size. Whoever owns the larger wins.

The game can be played with smaller or bigger boards

<img src="image.png" width=200>

## Resources

- [Where to play](https://en.boardgamearena.com/gamepanel?game=strands)
- [Post about Advanced Strategy](https://boardgamegeek.com/thread/3114220/strands-strategy-primer)
- [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)