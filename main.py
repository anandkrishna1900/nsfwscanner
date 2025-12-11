import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import asyncio
import logging


# Load environment variables
load_dotenv()
TOKEN = os.getenv('TOKEN')
PREFIX = os.getenv('PREFIX', ';')

if not TOKEN:
    logger.error("TOKEN not found in environment variables!")
    raise ValueError("No TOKEN found in .env file")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# CRITICAL: Enable all necessary intents
intents = discord.Intents.default()
intents.message_content = True  # MUST BE ENABLED for attachments
intents.messages = True
intents.guilds = True
intents.members = True


# Create bot
bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    help_command=None
)


@bot.event
async def on_ready():
    logger.info(f'✅ {bot.user.name} has connected to Discord!')
    logger.info(f'📊 Bot is in {len(bot.guilds)} guilds')
    logger.info(f'🔐 Message Content Intent: {intents.message_content}')


# CRITICAL: This allows on_message listeners in cogs to work
@bot.event
async def on_message(message):
    # Don't process bot messages
    if message.author.bot:
        return
    
    # Process commands first (this allows ; commands to work)
    await bot.process_commands(message)


# Load all cogs
async def load_cogs():
    cogs = [
        'cogs.info',
        'cogs.utility',
        'cogs.fun',
        'cogs.moderation',
        'cogs.automod',
        'cogs.automod_setup',
        'cogs.tasks',
        'cogs.errors'  # Global error handling
    ]
    for cog in cogs:
        try:
            await bot.load_extension(cog)
            logger.info(f'✅ Loaded {cog}')
        except Exception as e:
            logger.error(f'❌ Failed to load {cog}: {e}')


async def main():
    import time
    from database import init_db_pool, close_db
    
    bot.start_time = time.time()
    
    # Initialize Database
    try:
        await init_db_pool()
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        return

    async with bot:
        await load_cogs()
        try:
            await bot.start(TOKEN)
        except KeyboardInterrupt:
            # Handle user interrupt
            pass
        finally:
            await close_db()
            logger.info("Bot shutdown complete.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
