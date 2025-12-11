import discord
from discord.ext import commands

class Utility(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # Remove the duplicate avatar and serverinfo commands since they exist in info.py
    # You can add other utility commands here that don't conflict with other cogs
    
    @commands.hybrid_command(description="Check the bot's latency.")
    async def ping(self, ctx):
        """Check the bot's latency"""
        latency = round(self.bot.latency * 1000)
        em = discord.Embed(
            title="🏓 Pong!", 
            description=f"Bot latency: {latency}ms",
            color=discord.Color.green()
        )
        await ctx.send(embed=em)

    @commands.hybrid_command(description="Show how long the bot has been running.")
    async def uptime(self, ctx):
        """Show how long the bot has been running"""
        import time
        uptime_seconds = int(time.time() - self.bot.start_time) if hasattr(self.bot, 'start_time') else 0
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        em = discord.Embed(
            title="⏰ Bot Uptime",
            description=f"{hours}h {minutes}m {seconds}s",
            color=discord.Color.blue()
        )
        await ctx.send(embed=em)

    @commands.command(name="sync", hidden=True)
    @commands.is_owner()
    async def sync(self, ctx):
        """Sync commands globally"""
        msg = await ctx.send("🔄 Syncing commands...")
        try:
            synced = await self.bot.tree.sync()
            await msg.edit(content=f"✅ Synced {len(synced)} commands globally!")
        except Exception as e:
            await msg.edit(content=f"❌ Failed to sync: {e}")

async def setup(bot):
    await bot.add_cog(Utility(bot))
