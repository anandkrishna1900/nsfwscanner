# cogs/moderation.py
import discord
from discord.ext import commands
from database import add_modlog, add_scheduled, clear_warns, get_modlogs
from datetime import datetime, timezone, timedelta

class Moderation(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # --- Ban / Tempban / Unban ---
    @commands.hybrid_command(description="Ban a member from the server.")
    @commands.has_permissions(ban_members=True)
    async def ban(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        await member.ban(reason=reason)
        case = await add_modlog(member.id, "Ban", reason, ctx.author.id)
        em = discord.Embed(title="🔨 Banned", color=discord.Color.red())
        em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)
        try:
            await member.send(embed=discord.Embed(title="You were banned",
                                                  description=f"Server: {ctx.guild.name}\nReason: {reason}",
                                                  color=discord.Color.red()))
        except:
            pass

    @commands.hybrid_command(description="Temporarily ban a member.")
    @commands.has_permissions(ban_members=True)
    async def tempban(self, ctx, member: discord.Member, duration: str, *, reason: str = "No reason provided"):
        unit = duration[-1]
        try:
            amount = int(duration[:-1])
        except:
            return await ctx.send(embed=discord.Embed(description="Invalid duration format. Use `10m`, `2h`, `1d`.",
                                                      color=discord.Color.red()), ephemeral=True)
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        if unit not in multiplier:
            return await ctx.send(embed=discord.Embed(description="Invalid unit; use s/m/h/d.", color=discord.Color.red()), ephemeral=True)
        seconds = amount * multiplier[unit]

        await member.ban(reason=reason)
        case = await add_modlog(member.id, f"Tempban ({duration})", reason, ctx.author.id)

        execute_ts = int(datetime.now(timezone.utc).timestamp()) + seconds
        await add_scheduled(ctx.guild.id, member.id, "unban", execute_ts, None)

        em = discord.Embed(title="⏳ Tempban", color=discord.Color.orange())
        em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
        em.add_field(name="Duration", value=duration, inline=True)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)

    @commands.hybrid_command(description="Unban a user.")
    @commands.has_permissions(ban_members=True)
    async def unban(self, ctx, user_id: str, *, reason: str = "No reason provided"):
        # Hybrid commands pass arguments as strings often in slash commands if not typed strictly
        try:
            user_id = int(str(user_id)) # Handle mention or ID string
            user = await self.bot.fetch_user(user_id)
        except:
             return await ctx.send("Invalid user ID.", ephemeral=True)

        await ctx.guild.unban(user, reason=reason)
        case = await add_modlog(user_id, "Unban", reason, ctx.author.id)
        em = discord.Embed(title="✅ Unbanned", color=discord.Color.green())
        em.add_field(name="User", value=f"{user} ({user.id})", inline=False)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)

    # --- Kick ---
    @commands.hybrid_command(description="Kick a member from the server.")
    @commands.has_permissions(kick_members=True)
    async def kick(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        await member.kick(reason=reason)
        case = await add_modlog(member.id, "Kick", reason, ctx.author.id)
        em = discord.Embed(title="👢 Kicked", color=discord.Color.orange())
        em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)

    # --- Mute / Unmute ---
    @commands.hybrid_command(description="Mute a member (Timeout or Role).")
    @commands.has_permissions(manage_roles=True)
    async def mute(self, ctx, member: discord.Member, duration: str = None, *, reason: str = "No reason provided"):
        guild = ctx.guild
        role = discord.utils.get(guild.roles, name="Muted")
        if not role:
            role = await guild.create_role(name="Muted")
            for ch in guild.channels:
                try:
                    await ch.set_permissions(role, send_messages=False, speak=False, add_reactions=False)
                except:
                    pass
        await member.add_roles(role, reason=reason)

        if duration:
            unit = duration[-1]
            try:
                amount = int(duration[:-1])
            except:
                 return await ctx.send(embed=discord.Embed(description="Invalid duration format. Use `10m`, `2h`, `1d`.",
                                                      color=discord.Color.red()), ephemeral=True)
            multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            if unit not in multiplier:
                 return await ctx.send(embed=discord.Embed(description="Invalid unit; use s/m/h/d.", color=discord.Color.red()), ephemeral=True)
            seconds = amount * multiplier[unit]

            case = await add_modlog(member.id, f"Mute ({duration})", reason, ctx.author.id)
            execute_ts = int(datetime.now(timezone.utc).timestamp()) + seconds
            await add_scheduled(ctx.guild.id, member.id, "auto-unmute", execute_ts, None)

            em = discord.Embed(title="🔇 Muted", color=discord.Color.dark_gray())
            em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
            em.add_field(name="Duration", value=duration, inline=True)
            em.add_field(name="Reason", value=reason, inline=False)
            em.add_field(name="Case", value=f"#{case}", inline=False)
            await ctx.send(embed=em)
        else:
            case = await add_modlog(member.id, "Mute (Indefinite)", reason, ctx.author.id)
            em = discord.Embed(title="🔇 Muted", color=discord.Color.dark_gray())
            em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
            em.add_field(name="Reason", value=reason, inline=False)
            em.add_field(name="Case", value=f"#{case}", inline=False)
            await ctx.send(embed=em)

    @commands.hybrid_command(description="Unmute a member.")
    @commands.has_permissions(manage_roles=True)
    async def unmute(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        role = discord.utils.get(ctx.guild.roles, name="Muted")
        if role and role in member.roles:
            await member.remove_roles(role, reason=reason)
            case = await add_modlog(member.id, "Unmute", reason, ctx.author.id)
            em = discord.Embed(title="🔈 Unmuted", color=discord.Color.green())
            em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
            em.add_field(name="Reason", value=reason, inline=False)
            em.add_field(name="Case", value=f"#{case}", inline=False)
            await ctx.send(embed=em)
        else:
            await ctx.send(embed=discord.Embed(description="User is not muted.", color=discord.Color.red()), ephemeral=True)

    # --- Timeout ---
    @commands.hybrid_command(description="Timeout a member.")
    @commands.has_permissions(moderate_members=True)
    async def timeout(self, ctx, member: discord.Member, duration: str, *, reason: str = "No reason provided"):
        unit = duration[-1]
        try:
            amount = int(duration[:-1])
        except:
             return await ctx.send(embed=discord.Embed(description="Invalid duration format. Use `10m`, `2h`, `1d`.",
                                                      color=discord.Color.red()), ephemeral=True)
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        if unit not in multiplier:
             return await ctx.send(embed=discord.Embed(description="Invalid unit; use s/m/h/d.", color=discord.Color.red()), ephemeral=True)
        seconds = amount * multiplier[unit]

        until = discord.utils.utcnow() + timedelta(seconds=seconds)
        await member.timeout(until, reason=reason)

        case = await add_modlog(member.id, f"Timeout ({duration})", reason, ctx.author.id)
        execute_ts = int(datetime.now(timezone.utc).timestamp()) + seconds
        await add_scheduled(ctx.guild.id, member.id, "untimeout", execute_ts, None)

        em = discord.Embed(title="⏱️ Timeout", color=discord.Color.yellow())
        em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
        em.add_field(name="Duration", value=duration, inline=True)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)

    # --- Warn / Warnings ---
    @commands.hybrid_command(description="Warn a member.")
    @commands.has_permissions(manage_messages=True)
    async def warn(self, ctx, member: discord.Member, *, reason: str = "No reason provided"):
        case = await add_modlog(member.id, "Warn", reason, ctx.author.id)
        em = discord.Embed(title="⚠️ Warned", color=discord.Color.gold())
        em.add_field(name="User", value=f"{member} ({member.id})", inline=False)
        em.add_field(name="Reason", value=reason, inline=False)
        em.add_field(name="Case", value=f"#{case}", inline=False)
        await ctx.send(embed=em)

    @commands.hybrid_command(description="Check warnings for a member.")
    async def warnings(self, ctx, member: discord.Member):
        rows = await get_modlogs(member.id)
        warns = [r for r in rows if r[2] == "Warn"]
        if not warns:
            await ctx.send(embed=discord.Embed(description="No warnings found.", color=discord.Color.blurple()), ephemeral=True)
            return
        em = discord.Embed(title=f"⚠️ Warnings for {member}", color=discord.Color.gold())
        for case_id, user_id, action, reason, mod_id, ts in warns:
            try:
                moderator = await self.bot.fetch_user(mod_id)
            except:
                moderator = "Unknown"
            em.add_field(name=f"Case #{case_id} - {ts}", value=f"Reason: {reason}\nBy: {moderator}", inline=False)
        await ctx.send(embed=em)

    @commands.hybrid_command(description="Clear all warnings for a member.")
    @commands.has_permissions(administrator=True)
    async def clearwarns(self, ctx, member: discord.Member):
        await clear_warns(member.id)
        await ctx.send(embed=discord.Embed(description=f"Cleared warnings for {member}.", color=discord.Color.green()))

# --- ASYNC SETUP ---
async def setup(bot):
    await bot.add_cog(Moderation(bot))
