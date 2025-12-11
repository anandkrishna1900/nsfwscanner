# cogs/info.py
import discord
from discord.ext import commands
from database import get_modlogs, get_case
from datetime import datetime

class Info(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(name="help", description="Show help for the bot.")
    async def help_command(self, ctx, category: str = None):
        """Custom help command"""
        if not category:
            # Main help embed
            em = discord.Embed(
                title="🤖 Bot Commands",
                description="Use `/help <category>` for detailed commands",
                color=discord.Color.blue()
            )
            em.add_field(
                name="📋 Categories",
                value="`info` - Information commands\n"
                      "`fun` - Fun commands\n"
                      "`moderation` - Moderation commands\n"
                      "`utility` - Utility commands\n"
                      "`scanner` - Media scanner commands",
                inline=False
            )
            em.set_footer(text="Example: /help info")
            await ctx.send(embed=em)
            
        elif category.lower() == "info":
            em = discord.Embed(title="📋 Info Commands", color=discord.Color.blue())
            em.add_field(
                name="Commands",
                value="`modlogs <user>` - View user's moderation logs\n"
                      "`case <id>` - View specific case details\n"
                      "`userinfo [member]` - Show user information\n"
                      "`serverinfo` - Show server information\n"
                      "`avatar [member]` - Show user's avatar",
                inline=False
            )
            await ctx.send(embed=em)
            
        elif category.lower() == "fun":
            em = discord.Embed(title="🎮 Fun Commands", color=discord.Color.green())
            em.add_field(
                name="Commands",
                value="`roll [sides]` - Roll a dice (default 6 sides)\n"
                      "`coin` - Flip a coin\n"
                      "`joke` - Get a random joke",
                inline=False
            )
            await ctx.send(embed=em)
            
        elif category.lower() == "moderation":
            em = discord.Embed(title="🛡️ Moderation Commands", color=discord.Color.red())
            em.add_field(
                name="Commands",
                value="`ban <member> [reason]` - Ban a member\n"
                      "`tempban <member> <duration> [reason]` - Temporarily ban\n"
                      "`unban <user_id> [reason]` - Unban a user\n"
                      "`kick <member> [reason]` - Kick a member\n"
                      "`mute <member> [duration] [reason]` - Mute a member\n"
                      "`unmute <member> [reason]` - Unmute a member\n"
                      "`timeout <member> <duration> [reason]` - Timeout member\n"
                      "`warn <member> [reason]` - Warn a member\n"
                      "`warnings <member>` - View member's warnings\n"
                      "`clearwarns <member>` - Clear member's warnings",
                inline=False
            )
            await ctx.send(embed=em)
            
        elif category.lower() == "utility":
            em = discord.Embed(title="🔧 Utility Commands", color=discord.Color.purple())
            em.add_field(
                name="Commands",
                value="`ping` - Check bot latency\n"
                      "`uptime` - Show bot uptime",
                inline=False
            )
            await ctx.send(embed=em)
            
        elif category.lower() in ["scanner", "automod"]:
            em = discord.Embed(title="📸 Media Scanner Commands", color=discord.Color.orange())
            em.add_field(
                name="Setup Commands",
                value="`scanner` - Check scanner status\n"
                      "`scanner toggle` - Enable/disable scanner\n"
                      "`scanner punishment <type>` - Set punishment (kick/ban/timeout)\n"
                      "`scanner threshold <1-100>` - Set detection sensitivity\n"
                      "`scanner duration <minutes>` - Set timeout duration\n"
                      "`scanner logchannel [#channel]` - Set log channel\n"
                      "`scanner test` - Test API connection\n"
                      "`scanner formats` - Show supported file types",
                inline=False
            )
            em.add_field(
                name="What it scans",
                value="🖼️ **Images**: JPG, PNG, GIF, WebP, HEIC, etc.\n"
                      "🎞️ **Animations**: APNG, GIF, GIFV\n"
                      "📹 **Videos**: MP4, AVI, MOV, MKV, WebM, etc.\n"
                      "📸 **Raw Photos**: DNG, CR2, NEF, ARW, etc.",
                inline=False
            )
            em.add_field(
                name="Detection",
                value="🔍 Uses AI to detect NSFW and gore content\n"
                      "⚡ Supports 30+ file formats\n"
                      "🎯 Configurable sensitivity threshold\n"
                      "📝 Automatic logging and user notifications",
                inline=False
            )
            await ctx.send(embed=em)
            
        else:
            await ctx.send(embed=discord.Embed(
                description="❌ Invalid category! Use `/help` to see available categories.",
                color=discord.Color.red()
            ), ephemeral=True)

    @commands.hybrid_command(description="View modlogs for a user.")
    async def modlogs(self, ctx, user: discord.User):
        rows = await get_modlogs(user.id)
        if not rows:
            await ctx.send(embed=discord.Embed(description="No logs found.", color=discord.Color.greyple()), ephemeral=True)
            return

        em = discord.Embed(title=f"📜 Modlogs for {user}", color=discord.Color.blurple())
        for case_id, user_id, action, reason, mod_id, ts in rows:
            try:
                moderator = await self.bot.fetch_user(mod_id)
            except:
                moderator = "Unknown"
            em.add_field(name=f"Case #{case_id} | {action}", value=f"Reason: {reason}\nBy: {moderator}\nAt: {ts}", inline=False)

        await ctx.send(embed=em)

    @commands.hybrid_command(description="View details of a specific moderation case.")
    async def case(self, ctx, case_id: int):
        row = await get_case(case_id)
        if not row:
            return await ctx.send(embed=discord.Embed(description="Case not found.", color=discord.Color.red()), ephemeral=True)

        cid, user_id, action, reason, mod_id, ts = row
        try:
            user = await self.bot.fetch_user(user_id)
        except:
            user = f"ID {user_id}"
        try:
            moderator = await self.bot.fetch_user(mod_id)
        except:
            moderator = f"ID {mod_id}"

        em = discord.Embed(title=f"Case #{cid}", color=discord.Color.gold())
        em.add_field(name="User", value=f"{user} ({user_id})", inline=False)
        em.add_field(name="Action", value=action, inline=False)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Moderator", value=f"{moderator} ({mod_id})", inline=False)
        em.add_field(name="When", value=str(ts), inline=False)

        await ctx.send(embed=em)

    @commands.hybrid_command(description="Show user information.")
    async def userinfo(self, ctx, member: discord.Member = None):
        member = member or ctx.author
        roles = [r.mention for r in member.roles if r != ctx.guild.default_role]

        em = discord.Embed(title=f"ℹ️ User Info - {member}", color=discord.Color.blue())
        em.set_thumbnail(url=member.avatar.url if member.avatar else member.default_avatar.url)
        em.add_field(name="ID", value=member.id, inline=False)
        em.add_field(name="Account Created", value=member.created_at.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        em.add_field(name="Joined Server", value=member.joined_at.strftime("%Y-%m-%d %H:%M:%S") if member.joined_at else "Unknown", inline=False)
        em.add_field(name="Top Role", value=member.top_role.mention if member.top_role else "None", inline=False)
        em.add_field(name="Roles", value=", ".join(roles) if roles else "None", inline=False)
        em.add_field(name="Status", value=str(member.status).title(), inline=False)
        em.add_field(name="Bot?", value=str(member.bot), inline=False)

        await ctx.send(embed=em)

    @commands.hybrid_command(description="Show server information.")
    async def serverinfo(self, ctx):
        guild = ctx.guild
        em = discord.Embed(title=f"🏰 Server Info - {guild.name}", color=discord.Color.green())
        em.set_thumbnail(url=guild.icon.url if guild.icon else discord.Embed.Empty)
        em.add_field(name="Server ID", value=guild.id, inline=False)
        em.add_field(name="Owner", value=str(guild.owner), inline=False)
        em.add_field(name="Created On", value=guild.created_at.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        em.add_field(name="Members", value=guild.member_count, inline=False)
        em.add_field(name="Roles", value=len(guild.roles), inline=False)
        em.add_field(name="Text Channels", value=len(guild.text_channels), inline=True)
        em.add_field(name="Voice Channels", value=len(guild.voice_channels), inline=True)
        em.add_field(name="Boost Level", value=guild.premium_tier, inline=False)

        await ctx.send(embed=em)

    @commands.hybrid_command(description="Show user's avatar.")
    async def avatar(self, ctx, member: discord.Member = None):
        member = member or ctx.author
        em = discord.Embed(title=f"🖼️ Avatar - {member}", color=discord.Color.blurple())
        em.set_image(url=member.avatar.url if member.avatar else member.default_avatar.url)
        await ctx.send(embed=em)

# Async setup
async def setup(bot):
    await bot.add_cog(Info(bot))
