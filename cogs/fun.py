import discord
from discord.ext import commands
import random

class Fun(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(description="Roll a die (default 6 sides).")
    async def roll(self, ctx, sides: int = 6):
        """Roll a dice"""
        result = random.randint(1, sides)
        await ctx.send(f"🎲 You rolled: {result}")

    @commands.hybrid_command(description="Flip a coin.")
    async def coin(self, ctx):
        """Flip a coin"""
        await ctx.send(f"🪙 {'Heads' if random.choice([True, False]) else 'Tails'}")

    @commands.hybrid_command(description="Tell a random joke.")
    async def joke(self, ctx):
        jokes = [
            "Why did the chicken cross the road? To get to the other side!",
            "I told my computer I needed a break, and it froze.",
            "Why don’t skeletons fight each other? They don’t have the guts."
        ]
        await ctx.send(random.choice(jokes))

async def setup(bot):
    await bot.add_cog(Fun(bot))
