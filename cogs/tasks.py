import discord
from discord.ext import commands, tasks
from database import get_due_scheduled, remove_scheduled_by_id, add_modlog
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class Tasks(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.check_scheduled_actions.start()

    def cog_unload(self):
        self.check_scheduled_actions.cancel()

    @tasks.loop(seconds=60)
    async def check_scheduled_actions(self):
        try:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            due_tasks = await get_due_scheduled(now_ts)

            for task in due_tasks:
                task_id, guild_id, user_id, action, exec_ts, extra = task
                
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    # Guild might be unavailable or bot left
                    # We might want to keep it or delete it. Deleting to avoid clogging.
                    await remove_scheduled_by_id(task_id)
                    continue

                try:
                    # Fetch member/user (use fetch_user for bans as they aren't in guild)
                    if action == "unban":
                        user = await self.bot.fetch_user(user_id)
                        await guild.unban(user, reason="Tempban expired")
                        await add_modlog(user_id, "Unban", "Tempban expired", self.bot.user.id)
                        logger.info(f"✅ Unbanned {user} in {guild.name}")

                    elif action == "auto-unmute":
                        member = guild.get_member(user_id)
                        if member:
                            role = discord.utils.get(guild.roles, name="Muted")
                            if role and role in member.roles:
                                await member.remove_roles(role, reason="Mute duration expired")
                                await add_modlog(user_id, "Unmute", "Mute duration expired", self.bot.user.id)
                                logger.info(f"✅ Unmuted {member} in {guild.name}")
                    
                    elif action == "untimeout":
                         # Discord handles timeout expiration automatically, 
                         # but we might want to log it or ensure it's removed if we manually set it.
                         pass

                except Exception as e:
                    logger.error(f"❌ Failed to process task {task_id} ({action}): {e}")
                
                # Always remove the task so it doesn't loop forever
                await remove_scheduled_by_id(task_id)

        except Exception as e:
            logger.error(f"❌ Error in task loop: {e}")

    @check_scheduled_actions.before_loop
    async def before_check(self):
        await self.bot.wait_until_ready()

async def setup(bot):
    await bot.add_cog(Tasks(bot))
