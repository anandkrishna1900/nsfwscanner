import discord
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)

class ErrorHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        # Ignore if the command has its own error handler
        if hasattr(ctx.command, 'on_error'):
            return

        # Get original error if needed (for cogs)
        error = getattr(error, 'original', error)

        # Ignore invalid commands
        if isinstance(error, commands.CommandNotFound):
            return

        if isinstance(error, commands.MissingPermissions):
            await ctx.send(embed=discord.Embed(
                description="❌ You don't have permission to do that!",
                color=discord.Color.red()
            ))
        
        elif isinstance(error, commands.BotMissingPermissions):
            await ctx.send(embed=discord.Embed(
                description="❌ I don't have the required permissions!",
                color=discord.Color.red()
            ))
            
        elif isinstance(error, commands.BadArgument):
            await ctx.send(embed=discord.Embed(
                description="❌ Invalid arguments provided!",
                color=discord.Color.red()
            ))
            
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(embed=discord.Embed(
                description=f"❌ Missing required argument: `{error.param.name}`",
                color=discord.Color.red()
            ))
            
        else:
            logger.error(f"Unhandled error: {error}")
            # Optionally notify developer
            # await ctx.send("An unexpected error occurred.")

async def setup(bot):
    await bot.add_cog(ErrorHandler(bot))
