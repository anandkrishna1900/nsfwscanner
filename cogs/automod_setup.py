import discord
from discord.ext import commands
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_duration(minutes):
    if minutes < 60:
        return f"{minutes} minutes"
    elif minutes < 1440:
        hours = minutes // 60
        return f"{hours} hour{'s' if hours > 1 else ''}"
    elif minutes < 10080:
        days = minutes // 1440
        return f"{days} day{'s' if days > 1 else ''}"
    else:
        weeks = minutes // 10080
        return f"{weeks} week{'s' if weeks > 1 else ''}"

def parse_duration(duration_str):
    """Parse duration string like '7d', '2h', '30m', '10min' to minutes"""
    duration_str = duration_str.lower().strip()
    
    # Pattern: number followed by unit (d, h, m, min, etc.)
    pattern = r'^(\d+)\s*(d|day|days|h|hr|hrs|hour|hours|m|min|mins|minute|minutes)$'
    match = re.match(pattern, duration_str)
    
    if not match:
        return None
    
    amount = int(match.group(1))
    unit = match.group(2)
    
    # Convert to minutes
    if unit in ['d', 'day', 'days']:
        return amount * 1440
    elif unit in ['h', 'hr', 'hrs', 'hour', 'hours']:
        return amount * 60
    elif unit in ['m', 'min', 'mins', 'minute', 'minutes']:
        return amount
    
    return None

class SetupView(discord.ui.View):
    """Interactive setup wizard for AutoMod"""
    def __init__(self, ctx, config_file="automod_config.json"):
        super().__init__(timeout=300)  # 5 minutes timeout
        self.ctx = ctx
        self.config_file = config_file
        self.setup_data = {
            "enabled": True,
            "log_channel": None,
            "punishment": "none",
            "timeout_duration": 10,
            "ban_duration": None,  # None = permanent
            "nsfw_threshold": 50,
            "whitelisted_roles": [],
            "whitelisted_users": []
        }
        self.current_step = 0
        self.message = None

    async def start(self):
        """Start the setup wizard"""
        await self.show_step_1_toggle()

    async def show_step_1_toggle(self):
        """Step 1: Enable/Disable scanner"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 1/7",
            description="**Enable NSFW Scanner?**\n\nThe scanner will automatically detect and remove NSFW content from your server.",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 1: Toggle Scanner")

        view = discord.ui.View(timeout=300)

        enable_btn = discord.ui.Button(label="Enable", style=discord.ButtonStyle.green, emoji="✅")
        disable_btn = discord.ui.Button(label="Disable", style=discord.ButtonStyle.red, emoji="❌")

        async def enable_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["enabled"] = True
            await interaction.response.defer()
            await self.show_step_2_log_channel()

        async def disable_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["enabled"] = False
            await interaction.response.defer()
            await self.show_step_2_log_channel()

        enable_btn.callback = enable_callback
        disable_btn.callback = disable_callback

        view.add_item(enable_btn)
        view.add_item(disable_btn)

        if self.message:
            await self.message.edit(embed=embed, view=view)
        else:
            self.message = await self.ctx.send(embed=embed, view=view)

    async def show_step_2_log_channel(self):
        """Step 2: Select log channel (must be NSFW)"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 2/7",
            description="**Select Log Channel**\n\nChoose where NSFW detection logs should be sent.\n⚠️ **This must be a NSFW text channel for safety!**\n\nUse the dropdown below to select a channel.",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 2: Log Channel")

        view = discord.ui.View(timeout=300)

        channel_select = discord.ui.ChannelSelect(
            placeholder="Select an NSFW text channel...",
            channel_types=[discord.ChannelType.text],
            min_values=1,
            max_values=1,
        )

        async def channel_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)

            selected_channel_id = int(interaction.data["values"][0])
            selected_channel = self.ctx.guild.get_channel(selected_channel_id)

            if not selected_channel.is_nsfw():
                return await interaction.response.send_message("❌ Please select a NSFW text channel for logs.", ephemeral=True)

            self.setup_data["log_channel"] = selected_channel_id
            await interaction.response.defer()
            await self.show_step_3_punishment()

        channel_select.callback = channel_callback
        view.add_item(channel_select)

        await self.message.edit(embed=embed, view=view)

    async def show_step_3_punishment(self):
        """Step 3: Select punishment type"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 3/7",
            description="**Select Punishment**\n\nWhat should happen when NSFW content is detected?\n\n**None** - Just delete and log\n**Timeout** - Temporary mute\n**Kick** - Remove from server\n**Ban** - Permanent/temporary ban",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 3: Punishment Type")

        view = discord.ui.View(timeout=300)

        none_btn = discord.ui.Button(label="None (Delete Only)", style=discord.ButtonStyle.gray, emoji="🗑️")
        timeout_btn = discord.ui.Button(label="Timeout", style=discord.ButtonStyle.primary, emoji="⏱️")
        kick_btn = discord.ui.Button(label="Kick", style=discord.ButtonStyle.danger, emoji="👢")
        ban_btn = discord.ui.Button(label="Ban", style=discord.ButtonStyle.danger, emoji="🔨")

        async def none_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["punishment"] = "none"
            await interaction.response.defer()
            await self.show_step_7_threshold()

        async def timeout_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["punishment"] = "timeout"
            await interaction.response.defer()
            await self.show_step_4_timeout_duration()

        async def kick_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["punishment"] = "kick"
            await interaction.response.defer()
            await self.show_step_7_threshold()

        async def ban_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["punishment"] = "ban"
            await interaction.response.defer()
            await self.show_step_5_ban_duration()

        none_btn.callback = none_callback
        timeout_btn.callback = timeout_callback
        kick_btn.callback = kick_callback
        ban_btn.callback = ban_callback

        view.add_item(none_btn)
        view.add_item(timeout_btn)
        view.add_item(kick_btn)
        view.add_item(ban_btn)

        await self.message.edit(embed=embed, view=view)

    async def show_step_4_timeout_duration(self):
        """Step 4: Select timeout duration"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 4/7",
            description="**Timeout Duration**\n\nHow long should the timeout last?",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 4: Timeout Duration")

        view = discord.ui.View(timeout=300)

        durations = [
            ("5 minutes", 5),
            ("10 minutes", 10),
            ("30 minutes", 30),
            ("1 hour", 60),
            ("6 hours", 360),
            ("1 day", 1440),
            ("1 week", 10080),
            ("Custom...", 0)  # 0 means prompt for custom
        ]

        async def back_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.show_step_3_punishment()

        back_btn = discord.ui.Button(label="Back", style=discord.ButtonStyle.gray, row=1)
        back_btn.callback = back_callback
        view.add_item(back_btn)

        for label, minutes in durations:
            btn = discord.ui.Button(label=label, style=discord.ButtonStyle.primary)

            async def duration_callback(interaction, mins=minutes):
                if interaction.user != self.ctx.author:
                    return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
                
                if mins == 0:  # Custom duration requested
                    await interaction.response.send_message(
                        "Please type the timeout duration (e.g., `10m`, `2h`, `1d`):",
                        ephemeral=True
                    )

                    def check(m):
                        return m.author == self.ctx.author and m.channel == self.ctx.channel

                    try:
                        msg = await self.ctx.bot.wait_for('message', check=check, timeout=60)
                        custom_mins = parse_duration(msg.content)
                        
                        if custom_mins is None:
                            return await self.ctx.send("❌ Invalid format! Use formats like `10m`, `2h`, `7d`. Setup cancelled.")
                        
                        if custom_mins <= 0:
                            return await self.ctx.send("❌ Duration must be positive. Setup cancelled.")
                        
                        self.setup_data["timeout_duration"] = custom_mins
                        await self.show_step_7_threshold()
                    except Exception:
                        return await self.ctx.send("❌ Timeout input cancelled or timed out.")
                else:
                    await interaction.response.defer()
                    self.setup_data["timeout_duration"] = mins
                    await self.show_step_7_threshold()

            btn.callback = duration_callback
            view.add_item(btn)

        await self.message.edit(embed=embed, view=view)

    async def show_step_5_ban_duration(self):
        """Step 5: Select ban duration"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 5/7",
            description="**Ban Duration**\n\nHow long should the ban last?",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 5: Ban Duration")

        view = discord.ui.View(timeout=300)

        permanent_btn = discord.ui.Button(label="Permanent", style=discord.ButtonStyle.danger, emoji="🔒")
        temp_7d_btn = discord.ui.Button(label="7 days (Temporary)", style=discord.ButtonStyle.primary, emoji="⏱️")
        custom_btn = discord.ui.Button(label="Custom...", style=discord.ButtonStyle.secondary)

        async def back_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.show_step_3_punishment()

        back_btn = discord.ui.Button(label="Back", style=discord.ButtonStyle.gray, row=1)
        back_btn.callback = back_callback
        view.add_item(back_btn)

        async def permanent_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["ban_duration"] = None  # None = permanent
            await interaction.response.defer()
            await self.show_step_7_threshold()

        async def temp_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            self.setup_data["ban_duration"] = 7  # 7 days
            await interaction.response.defer()
            await self.show_step_7_threshold()

        async def custom_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            
            await interaction.response.send_message(
                "Please type the ban duration in days (e.g., `7d`, `14d`, `30d`):",
                ephemeral=True
            )

            def check(m):
                return m.author == self.ctx.author and m.channel == self.ctx.channel

            try:
                msg = await self.ctx.bot.wait_for('message', check=check, timeout=60)
                
                # Parse as days (convert to days from minutes)
                custom_mins = parse_duration(msg.content)
                if custom_mins is None:
                    return await self.ctx.send("❌ Invalid format! Use formats like `7d`, `14d`. Setup cancelled.")
                
                custom_days = custom_mins // 1440  # Convert minutes to days
                if custom_days <= 0:
                    return await self.ctx.send("❌ Duration must be at least 1 day. Setup cancelled.")
                
                self.setup_data["ban_duration"] = custom_days
                await self.show_step_7_threshold()
            except Exception:
                return await self.ctx.send("❌ Ban duration input cancelled or timed out.")

        permanent_btn.callback = permanent_callback
        temp_7d_btn.callback = temp_callback
        custom_btn.callback = custom_callback

        view.add_item(permanent_btn)
        view.add_item(temp_7d_btn)
        view.add_item(custom_btn)

        await self.message.edit(embed=embed, view=view)

    async def show_step_7_threshold(self):
        """Step 7: Set NSFW detection threshold (custom numeric input)"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 6/7",
            description=(
                "**Set NSFW Detection Threshold**\n\n"
                "Enter a value between 1 and 100 representing the percentage NSFW detection threshold.\n"
                "Lower values are more sensitive and detect milder content.\n\n"
                "**Guidelines:**\n"
                "- Below 30: Very sensitive, detects mild NSFW content.\n"
                "- Around 50: Medium sensitivity (recommended).\n"
                "- Above 70: Very high, detects only strong NSFW content."
            ),
            color=discord.Color.blue(),
        )
        embed.set_footer(text="Step 6: NSFW Threshold")

        view = discord.ui.View(timeout=300)

        back_btn = discord.ui.Button(label="Back", style=discord.ButtonStyle.gray)

        async def back_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.show_step_3_punishment()

        async def wait_threshold_input(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)

            await interaction.response.send_message(
                "Please enter the NSFW detection threshold as a number between 1 and 100:",
                ephemeral=True
            )

            def check(m):
                return m.author == self.ctx.author and m.channel == self.ctx.channel and m.content.isdigit()

            try:
                msg = await self.ctx.bot.wait_for('message', check=check, timeout=60)
                val = int(msg.content)
                if not 1 <= val <= 100:
                    return await self.ctx.send("❌ Value must be between 1 and 100. Setup cancelled.")

                self.setup_data["nsfw_threshold"] = val
                await self.show_step_6_whitelist()
            except Exception:
                await self.ctx.send("❌ Input timed out or invalid. Setup cancelled.")

        input_btn = discord.ui.Button(label="Enter Threshold", style=discord.ButtonStyle.primary)
        input_btn.callback = wait_threshold_input
        back_btn.callback = back_callback

        view.add_item(input_btn)
        view.add_item(back_btn)

        if self.message:
            await self.message.edit(embed=embed, view=view)
        else:
            self.message = await self.ctx.send(embed=embed, view=view)

    async def show_step_6_whitelist(self):
        """Step 6: Whitelist roles/users"""
        embed = discord.Embed(
            title="🤖 AutoMod Setup Wizard - Step 7/7",
            description="**Whitelist Roles/Users**\n\nDo you want to whitelist any roles or users from NSFW checks?\n\nWhitelisted members won't be scanned.",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Step 7: Whitelist (Optional)")

        view = discord.ui.View(timeout=300)

        role_select = discord.ui.RoleSelect(
            placeholder="Select roles to whitelist (optional)...",
            min_values=0,
            max_values=10
        )

        skip_btn = discord.ui.Button(label="Skip / No Whitelist", style=discord.ButtonStyle.gray, emoji="⏭️", row=1)
        done_btn = discord.ui.Button(label="Done (Review Settings)", style=discord.ButtonStyle.green, emoji="✅", row=1)

        async def role_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)

            selected_roles = [int(role_id) for role_id in interaction.data.get("values", [])]
            self.setup_data["whitelisted_roles"] = selected_roles

            await interaction.response.send_message(
                f"✅ Whitelisted {len(selected_roles)} role(s). Click 'Done' to review settings.",
                ephemeral=True
            )

        async def skip_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.show_preview()

        async def done_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.show_preview()

        role_select.callback = role_callback
        skip_btn.callback = skip_callback
        done_btn.callback = done_callback

        view.add_item(role_select)
        view.add_item(skip_btn)
        view.add_item(done_btn)

        await self.message.edit(embed=embed, view=view)

    async def show_preview(self):
        """Show preview and confirmation"""
        log_channel = self.ctx.guild.get_channel(self.setup_data["log_channel"]) if self.setup_data["log_channel"] else None
        whitelisted_roles = [
            self.ctx.guild.get_role(role_id).mention
            for role_id in self.setup_data["whitelisted_roles"]
            if self.ctx.guild.get_role(role_id)
        ]

        embed = discord.Embed(
            title="🤖 AutoMod Setup Preview",
            description="**Review your settings before confirming:**",
            color=discord.Color.green()
        )
        status = "✅ Enabled" if self.setup_data["enabled"] else "❌ Disabled"
        embed.add_field(name="Scanner Status", value=status, inline=True)
        embed.add_field(name="Log Channel", value=log_channel.mention if log_channel else "Not set", inline=True)

        ban_duration = self.setup_data['ban_duration']
        ban_text = "Permanent" if ban_duration is None else f"{ban_duration} days"
        punishment_display = {
            "none": "🗑️ Delete Only",
            "timeout": f"⏱️ Timeout ({format_duration(self.setup_data['timeout_duration'])})",
            "kick": "👢 Kick",
            "ban": f"🔨 Ban ({ban_text})"
        }
        embed.add_field(
            name="Punishment",
            value=punishment_display.get(self.setup_data["punishment"], "Unknown"),
            inline=True
        )
        embed.add_field(name="Detection Threshold", value=f"{self.setup_data['nsfw_threshold']}%", inline=True)
        if whitelisted_roles:
            embed.add_field(
                name=f"Whitelisted Roles ({len(whitelisted_roles)})",
                value="\n".join(whitelisted_roles[:5]) + ("\n..." if len(whitelisted_roles) > 5 else ""),
                inline=False
            )
        else:
            embed.add_field(name="Whitelist", value="None", inline=False)

        embed.set_footer(text="Click 'Confirm' to save these settings")

        view = discord.ui.View(timeout=300)

        confirm_btn = discord.ui.Button(label="Confirm & Save", style=discord.ButtonStyle.green, emoji="✅")
        cancel_btn = discord.ui.Button(label="Cancel", style=discord.ButtonStyle.red, emoji="❌")

        async def confirm_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            await interaction.response.defer()
            await self.save_config()

        async def cancel_callback(interaction):
            if interaction.user != self.ctx.author:
                return await interaction.response.send_message("❌ Only the command author can use this!", ephemeral=True)
            cancel_embed = discord.Embed(
                title="❌ Setup Cancelled",
                description="AutoMod setup has been cancelled. No changes were made.",
                color=discord.Color.red()
            )
            await interaction.response.edit_message(embed=cancel_embed, view=None)

        confirm_btn.callback = confirm_callback
        cancel_btn.callback = cancel_callback

        view.add_item(confirm_btn)
        view.add_item(cancel_btn)

        await self.message.edit(embed=embed, view=view)

    async def save_config(self):
        """Save configuration to file"""
        try:
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
            except FileNotFoundError:
                config = {}
            guild_id = str(self.ctx.guild.id)
            config[guild_id] = self.setup_data

            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)

            # RELOAD the automod config so it uses the new settings immediately
            automod_cog = self.ctx.bot.get_cog("AutoMod")
            if automod_cog:
                automod_cog.load_config()  # Reload the config
                logger.info("🔄 AutoMod config reloaded")

            log_channel = self.ctx.guild.get_channel(self.setup_data["log_channel"])

            success_embed = discord.Embed(
                title="✅ AutoMod Setup Complete!",
                description=f"NSFW scanner has been configured successfully.\n\nLogs will be sent to {log_channel.mention if log_channel else 'Not set'}",
                color=discord.Color.green()
            )
            success_embed.add_field(
                name="Next Steps",
                value="• Make sure the scanner API is running (`scanner_api.py`)\n• Test with `;scanner test`\n• Upload a test image to verify detection",
                inline=False
            )

            await self.message.edit(embed=success_embed, view=None)

            logger.info(f"✅ AutoMod configured for guild {self.ctx.guild.name}")

        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Setup Failed",
                description=f"Failed to save configuration: {e}",
                color=discord.Color.red()
            )
            await self.message.edit(embed=error_embed, view=None)
            logger.error(f"Failed to save config: {e}")

class AutoModSetup(commands.Cog):
    """Interactive setup wizard for AutoMod"""
    def __init__(self, bot):
        self.bot = bot
    
    @commands.hybrid_command(name="setup", description="Interactive setup wizard for NSFW scanner.")
    @commands.has_permissions(administrator=True)
    async def scanner_setup(self, ctx):
        """Interactive setup wizard for NSFW scanner"""
        view = SetupView(ctx)
        await view.start()

async def setup(bot):
    await bot.add_cog(AutoModSetup(bot))
