import discord
from discord.ext import commands
import aiohttp
import json
from datetime import datetime, timedelta
import logging
import re
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoveTimeoutView(discord.ui.View):
    """Button view to remove timeout"""
    def __init__(self, user_id: int):
        super().__init__(timeout=None)
        self.user_id = user_id
    
    @discord.ui.button(label="Remove Timeout", style=discord.ButtonStyle.green, emoji="✅")
    async def remove_timeout_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            # Check if user has permissions
            if not interaction.user.guild_permissions.moderate_members:
                return await interaction.response.send_message("❌ You don't have permission to do this!", ephemeral=True)
            
            # Get the member
            member = interaction.guild.get_member(self.user_id)
            if not member:
                return await interaction.response.send_message("❌ User not found in server!", ephemeral=True)
            
            # Remove timeout
            await member.timeout(None, reason=f"Timeout removed by {interaction.user.name}")
            
            # Update the embed to show timeout was removed
            original_embed = interaction.message.embeds[0]
            
            # Add new field showing timeout was removed
            original_embed.add_field(
                name="⚠️ Timeout Status",
                value=f"**Removed by:** {interaction.user.mention}\n**At:** <t:{int(discord.utils.utcnow().timestamp())}:F>",
                inline=False
            )
            
            # Change embed color to show it's been handled
            original_embed.color = 0x808080  # Gray color
            
            # Update button to disabled
            button.disabled = True
            button.label = "Timeout Removed"
            button.style = discord.ButtonStyle.gray
            
            await interaction.response.edit_message(embed=original_embed, view=self)
            
            # Send confirmation
            await interaction.followup.send(f"✅ Removed timeout from {member.mention}", ephemeral=True)
            
            logger.info(f"✅ {interaction.user.name} removed timeout from {member.name}")
            
        except Exception as e:
            logger.error(f"Failed to remove timeout: {e}")
            await interaction.response.send_message(f"❌ Failed to remove timeout: {e}", ephemeral=True)

class AutoMod(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config_file = "automod_config.json"
        self.load_config()
        self.api_endpoint = "http://127.0.0.1:8000/v1/detect/urls"
        logger.info("🤖 AutoMod initialized with NSFW Detection AI")
    
    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
    
    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def get_server_config(self, guild_id):
        guild_id = str(guild_id)
        if guild_id not in self.config:
            self.config[guild_id] = {
                "enabled": True,
                "punishment": "timeout",
                "timeout_duration": 10,
                "ban_duration": None,
                "nsfw_threshold": 50,
                "log_channel": None,
                "whitelisted_roles": [],
                "whitelisted_users": []
            }
            self.save_config()
        
        default_keys = {
            "enabled": True,
            "punishment": "timeout",
            "timeout_duration": 10,
            "ban_duration": None,
            "nsfw_threshold": 50,
            "log_channel": None,
            "whitelisted_roles": [],
            "whitelisted_users": []
        }
        
        for key, default_value in default_keys.items():
            if key not in self.config[guild_id]:
                self.config[guild_id][key] = default_value
        
        self.save_config()
        return self.config[guild_id]
    
    def extract_image_urls(self, message):
        """Extract image URLs from attachments, embeds, and message content"""
        urls = []
        sources = []
        file_info = []  # Store (filename, url, size)
        
        # Check attachments
        for att in message.attachments:
            urls.append(att.url)
            sources.append(f"Attachment: {att.filename}")
            file_info.append((att.filename, att.url, att.size))
            logger.info(f"   📎 Attachment: {att.filename}")
        
        # Check embeds
        for embed in message.embeds:
            if embed.image and embed.image.url:
                urls.append(embed.image.url)
                sources.append(f"Embed Image")
                file_info.append(("embed_image", embed.image.url, 0))
                logger.info(f"   🖼️ Embed Image: {embed.image.url[:50]}...")
            
            if embed.thumbnail and embed.thumbnail.url:
                urls.append(embed.thumbnail.url)
                sources.append(f"Embed Thumbnail")
                file_info.append(("embed_thumbnail", embed.thumbnail.url, 0))
                logger.info(f"   🖼️ Embed Thumbnail")
        
        # Check message content for URLs
        if message.content:
            image_patterns = [
                r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp)',
                r'https?://cdn\.discordapp\.com/attachments/[^\s]+',
                r'https?://media\.discordapp\.net/attachments/[^\s]+',
                r'https?://i\.imgur\.com/[^\s]+',
            ]
            
            for pattern in image_patterns:
                matches = re.findall(pattern, message.content, re.IGNORECASE)
                for match in matches:
                    clean_url = match.split('?')[0]
                    if clean_url not in urls:
                        urls.append(clean_url)
                        sources.append(f"URL in message")
                        file_info.append(("linked_image", clean_url, 0))
                        logger.info(f"   🔗 URL: {clean_url[:50]}...")
        
        return urls, sources, file_info
    
    async def check_image_content(self, image_urls):
        """Check images using AI API"""
        logger.info(f"🔍 Sending {len(image_urls)} URLs to AI")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"urls": image_urls}
                async with session.post(
                    self.api_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        for i, result in enumerate(results):
                            status = "🚨 NSFW" if result.get("is_nsfw") else "✅ Safe"
                            confidence = result.get("confidence_percentage", 0)
                            logger.info(f"   Result {i+1}: {status} ({confidence}%)")
                        return results
                    else:
                        logger.error(f"❌ API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"❌ Connection error: {e}", exc_info=True)
            return None
    
    async def log_action(self, user, action, reason):
        try:
            # Use database instead of JSON
            from database import add_modlog
            # moderator_id is None/0 for AI/AutoMod
            # For postgres, ensure user.id is int
            await add_modlog(user.id, action, reason, self.bot.user.id)
            logger.info(f"📝 Logged action to database: {action} for {user}")
        except Exception as e:
            logger.error(f"❌ Failed to log to database: {e}")
    
    async def send_log_to_channel(self, message, file_info, punishment_type, violations):
        """Send detailed log to configured log channel"""
        config = self.get_server_config(message.guild.id)
        log_channel_id = config.get("log_channel")
        
        logger.info(f"📝 Attempting to log. Channel ID: {log_channel_id}")
        
        if not log_channel_id:
            logger.warning("❌ No log channel configured")
            return
        
        try:
            log_channel = message.guild.get_channel(int(log_channel_id))
            if not log_channel:
                logger.error(f"❌ Log channel {log_channel_id} not found")
                return
            
            logger.info(f"✅ Found log channel: #{log_channel.name}")
            
            # Get member info
            member = message.author
            
            # Calculate times
            now = discord.utils.utcnow()
            
            # Get roles
            roles_list = [role.name for role in member.roles if role.name != "@everyone"]
            roles_text = ", ".join(roles_list) if roles_list else "None"
            
            # Create embed
            embed = discord.Embed(
                title="🚨 NSFW Content Detected",
                color=0x2B2D31,
                timestamp=now
            )
            
            embed.set_thumbnail(url=member.display_avatar.url)
            
            # Add fields
            embed.add_field(
                name="User:",
                value=f"{member.mention} (`{member.id}`)",
                inline=False
            )
            
            # Use channel mention instead of name
            embed.add_field(
                name="Channel:",
                value=message.channel.mention,
                inline=False
            )
            
            embed.add_field(
                name="Message ID:",
                value=f"`{message.id}`",
                inline=False
            )
            
            embed.add_field(
                name="👤 User Information",
                value=f"**Username:** {member.name}\n**Display Name:** {member.display_name}",
                inline=False
            )
            
            # Use Discord timestamps for account creation
            account_created_timestamp = int(member.created_at.timestamp())
            embed.add_field(
                name="📅 Account Information",
                value=f"**Created:** <t:{account_created_timestamp}:F>\n**Age:** <t:{account_created_timestamp}:R>",
                inline=True
            )
            
            # Use Discord timestamps for join date
            if member.joined_at:
                joined_timestamp = int(member.joined_at.timestamp())
                embed.add_field(
                    name="🏠 Server Information",
                    value=f"**Joined:** <t:{joined_timestamp}:F>\n**Member for:** <t:{joined_timestamp}:R>",
                    inline=True
                )
            else:
                embed.add_field(
                    name="🏠 Server Information",
                    value=f"**Joined:** Unknown\n**Member for:** Unknown",
                    inline=True
                )
            
            embed.add_field(
                name="🎭 Roles",
                value=roles_text,
                inline=False
            )
            
            message_text = message.content if message.content else "No text content"
            embed.add_field(
                name="💬 Message Content",
                value=message_text[:1024],
                inline=False
            )
            
            # Attachments with clickable URLs
            if file_info:
                attachments_list = []
                for filename, url, size in file_info:
                    if size > 0:
                        link_text = f"**[{filename}]({url})**"
                        attachments_list.append(f"{link_text} `({size} bytes)`")
                    else:
                        attachments_list.append(f"**[{filename}]({url})**")
                
                attachments_text = "\n".join(attachments_list)
                if len(attachments_text) > 1024:
                    attachments_text = attachments_text[:1000] + "\n...(truncated)"
                
                embed.add_field(
                    name="📎 Attachments",
                    value=attachments_text,
                    inline=False
                )
            
            # Add AI detection results with confidence percentages
            if violations:
                violations_text = "\n".join(violations)
                embed.add_field(
                    name="🤖 AI Detection Results",
                    value=violations_text[:1024],
                    inline=False
                )
            
            # Add the server's configured threshold
            threshold = config.get("nsfw_threshold", 50)
            embed.add_field(
                name="⚙️ Server Threshold",
                value=f"{threshold}%",
                inline=True
            )
            
            embed.set_footer(text="NSFW Detection System • Raiden")
            
            logger.info("📤 Sending log embed...")
            
            # Add button if punishment is timeout
            view = None
            if punishment_type == "timeout":
                view = RemoveTimeoutView(member.id)
            
            # Send the main embed first
            log_message = await log_channel.send(embed=embed, view=view)
            
            # Send image preview as separate message with spoiler (AFTER main log)
            if file_info:
                try:
                    # Download and re-upload the first image to preserve it
                    first_image_url = file_info[0][1]
                    first_filename = file_info[0][0]
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(first_image_url, timeout=10) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                
                                # Create file object with spoiler
                                file = discord.File(
                                    fp=BytesIO(image_data),
                                    filename=f"SPOILER_{first_filename}",
                                    spoiler=True
                                )
                                
                                # Send as separate message with spoiler warning
                                preview_embed = discord.Embed(
                                    description="⚠️ **Flagged Content Preview** (Click to reveal)",
                                    color=0xFF0000
                                )
                                
                                await log_channel.send(embed=preview_embed, file=file)
                                logger.info("✅ Image preview sent with spoiler")
                            else:
                                logger.warning(f"Failed to download image for preview: {resp.status}")
                except Exception as e:
                    logger.error(f"Failed to send image preview: {e}")
            
            logger.info("✅ Log sent successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to send log: {e}", exc_info=True)
    
    async def punish_user(self, message, reason, config):
        punishment = config["punishment"]
        duration = config.get("timeout_duration", 10)
        ban_duration = config.get("ban_duration")
        
        logger.info(f"⚡ {punishment} → {message.author.name}")
        
        try:
            if punishment == "kick":
                await message.author.kick(reason=f"AI: {reason}")
                await self.log_action(message.author, "Kick", reason)
            
            elif punishment == "ban":
                if ban_duration:
                    # Temporary ban (delete after X days)
                    await message.author.ban(reason=f"AI: {reason} (Temp: {ban_duration}d)", delete_message_days=0)
                else:
                    # Permanent ban
                    await message.author.ban(reason=f"AI: {reason}")
                await self.log_action(message.author, "Ban", reason)
            
            elif punishment == "timeout":
                await message.author.timeout(
                    discord.utils.utcnow() + timedelta(minutes=duration),
                    reason=f"AI: {reason}"
                )
                await self.log_action(message.author, "Timeout", reason)
            
            elif punishment == "none":
                # Just delete and log, no punishment
                await self.log_action(message.author, "Warning", reason)
        
        except Exception as e:
            logger.error(f"❌ Punishment failed: {e}")
    
    @commands.hybrid_group(invoke_without_command=True, name="scanner", description="NSFW Scanner configuration")
    @commands.has_permissions(manage_messages=True)
    async def scanner(self, ctx):
        """Scanner status and configuration"""
        if ctx.invoked_subcommand is None:
            config = self.get_server_config(ctx.guild.id)
            status = "🟢 Enabled" if config["enabled"] else "🔴 Disabled"
            
            # Get log channel
            log_channel = None
            if config["log_channel"]:
                log_channel = ctx.guild.get_channel(int(config["log_channel"]))
            
            em = discord.Embed(title="🤖 NSFW Scanner", color=discord.Color.blue())
            em.add_field(name="Status", value=status, inline=True)
            em.add_field(name="Threshold", value=f"{config['nsfw_threshold']}%", inline=True)
            em.add_field(name="Punishment", value=config["punishment"].title(), inline=True)
            
            if config["punishment"] == "timeout":
                minutes = config["timeout_duration"]
                if minutes >= 1440:
                    days = minutes // 1440
                    remaining = minutes % 1440
                    if remaining == 0:
                        duration_text = f"{days}d"
                    else:
                        hours = remaining // 60
                        duration_text = f"{days}d {hours}h"
                elif minutes >= 60:
                    hours = minutes // 60
                    remaining = minutes % 60
                    if remaining == 0:
                        duration_text = f"{hours}h"
                    else:
                        duration_text = f"{hours}h {remaining}m"
                else:
                    duration_text = f"{minutes}m"
                
                em.add_field(name="Duration", value=duration_text, inline=True)
            
            # Test API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:8000/", timeout=3) as resp:
                        api_status = "🟢 Online" if resp.status == 200 else "🔴 Error"
            except:
                api_status = "🔴 Offline"
            
            em.add_field(name="API", value=api_status, inline=True)
            
            # Log channel
            log_status = log_channel.mention if log_channel else "Not set"
            em.add_field(name="Log Channel", value=log_status, inline=True)
            
            # Whitelist info
            whitelisted_roles = config.get("whitelisted_roles", [])
            if whitelisted_roles:
                em.add_field(name="Whitelisted Roles", value=f"{len(whitelisted_roles)} role(s)", inline=True)
            
            em.add_field(name="Scans", value="Attachments, Embeds, URLs", inline=False)
            em.set_footer(text="Use /scanner commands to configure")
            
            await ctx.send(embed=em)
    
    @scanner.command(description="Enable/disable scanner")
    @commands.has_permissions(manage_messages=True)
    async def toggle(self, ctx):
        """Enable/disable scanner"""
        config = self.get_server_config(ctx.guild.id)
        config["enabled"] = not config["enabled"]
        self.save_config()
        status = "enabled" if config["enabled"] else "disabled"
        await ctx.send(f"✅ Scanner {status}")
    
    @scanner.command(description="Set detection threshold (1-100)")
    @commands.has_permissions(manage_messages=True)
    async def threshold(self, ctx, percentage: int):
        """Set detection threshold (1-100)"""
        if not 1 <= percentage <= 100:
            return await ctx.send("❌ Use 1-100", ephemeral=True)
        
        config = self.get_server_config(ctx.guild.id)
        config["nsfw_threshold"] = percentage
        self.save_config()
        await ctx.send(f"✅ Threshold: {percentage}%")
    
    @scanner.command(description="Set punishment type (none/kick/ban/timeout)")
    @commands.has_permissions(manage_messages=True)
    async def punishment(self, ctx, ptype: str):
        """Set punishment type (none/kick/ban/timeout)"""
        if ptype.lower() not in ["none", "kick", "ban", "timeout"]:
            return await ctx.send("❌ Use: none, kick, ban, timeout", ephemeral=True)
        
        config = self.get_server_config(ctx.guild.id)
        config["punishment"] = ptype.lower()
        self.save_config()
        await ctx.send(f"✅ Punishment: {ptype.lower()}")
    
    @scanner.command(description="Set timeout duration")
    @commands.has_permissions(manage_messages=True)
    async def duration(self, ctx, duration: str):
        """Set timeout duration (e.g., 10m, 2h, 1d, 30s)"""
        pattern = r'^(\d+)([smhd])$'
        match = re.match(pattern, duration.lower())
        
        if not match:
            return await ctx.send("❌ Invalid format! Use: `30s`, `10m`, `2h`, or `1d`", ephemeral=True)
        
        amount, unit = match.groups()
        amount = int(amount)
        
        if unit == 's':
            minutes = amount / 60
            if minutes < 1:
                minutes = 1
            else:
                minutes = int(minutes)
            display = f"{amount} seconds"
        elif unit == 'm':
            minutes = amount
            display = f"{amount} minutes"
        elif unit == 'h':
            minutes = amount * 60
            display = f"{amount} hours"
        elif unit == 'd':
            minutes = amount * 1440
            display = f"{amount} days"
        
        if minutes < 1:
            return await ctx.send("❌ Minimum duration is 1 minute", ephemeral=True)
        if minutes > 40320:
            return await ctx.send("❌ Maximum duration is 28 days", ephemeral=True)
        
        config = self.get_server_config(ctx.guild.id)
        config["timeout_duration"] = int(minutes)
        self.save_config()
        
        await ctx.send(f"✅ Timeout duration set to **{display}** ({int(minutes)} minutes)")
    
    @scanner.command(name="logchannel", description="Set log channel for NSFW detections")
    @commands.has_permissions(manage_messages=True)
    async def log_channel(self, ctx, channel: discord.TextChannel = None):
        """Set log channel for NSFW detections"""
        config = self.get_server_config(ctx.guild.id)
        
        if channel is None:
            config["log_channel"] = None
            self.save_config()
            await ctx.send("✅ Log channel disabled")
        else:
            config["log_channel"] = channel.id
            self.save_config()
            
            test_embed = discord.Embed(
                title="✅ Log Channel Configured",
                description=f"NSFW detection logs will be sent to {channel.mention}",
                color=discord.Color.green()
            )
            await ctx.send(embed=test_embed)
            
            welcome_embed = discord.Embed(
                title="🤖 NSFW Detection System",
                description="This channel will receive detailed logs when NSFW content is detected and removed.",
                color=discord.Color.blue()
            )
            welcome_embed.add_field(
                name="What's Logged:",
                value="• User information\n• Account details\n• Server join date\n• Message content\n• Image preview (spoiler)\n• Remove timeout button",
                inline=False
            )
            await channel.send(embed=welcome_embed)
    
    @scanner.command(description="Test API connection")
    @commands.has_permissions(manage_messages=True)
    async def test(self, ctx):
        """Test API connection"""
        await ctx.send("🔍 Testing API...", ephemeral=True)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:8000/", timeout=5) as resp:
                    if resp.status == 200:
                        await ctx.send("✅ API is online and working!", ephemeral=True)
                    else:
                        await ctx.send(f"⚠️ API returned status {resp.status}", ephemeral=True)
        except:
            await ctx.send("❌ API is offline. Make sure `scanner_api.py` is running!", ephemeral=True)
    
    @commands.Cog.listener()
    async def on_message(self, message):
        """Main scanner - monitors all messages for images"""
        try:
            if message.author.bot or not message.guild:
                return
            
            config = self.get_server_config(message.guild.id)
            if not config["enabled"]:
                return
            
            # Check whitelist - roles
            if config.get("whitelisted_roles"):
                member_role_ids = [role.id for role in message.author.roles]
                if any(role_id in config["whitelisted_roles"] for role_id in member_role_ids):
                    logger.info(f"✅ {message.author.name} is whitelisted (role)")
                    return
            
            # Check whitelist - users
            if message.author.id in config.get("whitelisted_users", []):
                logger.info(f"✅ {message.author.name} is whitelisted (user)")
                return
            
            urls, sources, file_info = self.extract_image_urls(message)
            if not urls:
                return
            
            logger.info(f"📎 {message.author.name} → {len(urls)} image(s) detected")
            
            results = await self.check_image_content(urls)
            if not results:
                return
            
            violations = []
            flagged_files = []
            
            for i, result in enumerate(results):
                if result.get("is_nsfw") and result.get("confidence_percentage", 0) >= config["nsfw_threshold"]:
                    violations.append(f"{sources[i]} ({result['confidence_percentage']}%)")
                    if i < len(file_info):
                        flagged_files.append(file_info[i])
            
            if violations:
                logger.info(f"🚨 FLAGGED: {len(violations)} violation(s)")
                
                # Log to channel BEFORE deleting (pass violations data)
                await self.send_log_to_channel(message, flagged_files, config["punishment"], violations)
                
                try:
                    await message.delete()
                    logger.info("🗑️ Deleted")
                except:
                    pass
                
                await self.punish_user(message, f"NSFW content detected", config)
                
                try:
                    em = discord.Embed(
                        title="🚨 Content Removed",
                        description=f"Your content in **{message.guild.name}** was flagged by AI",
                        color=discord.Color.red()
                    )
                    em.add_field(name="Reason", value="NSFW content detected", inline=False)
                    em.add_field(name="Action", value=f"{config['punishment'].title()}", inline=False)
                    await message.author.send(embed=em)
                except:
                    pass
            else:
                logger.info("✅ Clean")
        
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)

async def setup(bot):
    await bot.add_cog(AutoMod(bot))
