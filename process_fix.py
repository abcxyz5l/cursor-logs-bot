# This is the clean process_selected_actions function to replace the broken one
async def process_selected_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process all selected actions one by one."""
    selected = context.user_data.get("selected_actions", set())
    url = context.user_data.get("url")
    pwd = context.user_data.get("archive_password")
    keyword = context.user_data.get("combo_keyword")
    main_msg = context.user_data.get("main_message")
    
    if not url:
        await update.message.reply_text("No URL found. Please start over.")
        return
    
    # Download ONCE at the beginning
    if main_msg:
        await safe_edit(main_msg, "📥 Downloading file...")
    else:
        main_msg = await update.message.reply_text("📥 Downloading file...")
        context.user_data["main_message"] = main_msg
    
    dest_path = await download_from_context_url(update, context, progress_message=main_msg)
    if not dest_path or not os.path.exists(dest_path):
        await safe_edit(main_msg, "❌ Download failed or file not found.")
        context.user_data.clear()
        return
    
    await safe_edit(main_msg, f"✅ Downloaded successfully!\nProcessing actions...")
    
    # Now process each action with the same downloaded file
    for action in sorted(selected):
        try:
            if action == "unzip":
                await safe_edit(main_msg, "🔄 Processing: Unzip...\n✅ Extracting...")
                tmp_dir = tempfile.mkdtemp(prefix="unzip_", dir=RESULTS_DIR)
                try:
                    is_zip = zipfile.is_zipfile(dest_path)
                    if is_zip:
                        with zipfile.ZipFile(dest_path) as z:
                            pwd_bytes = pwd.encode("utf-8") if pwd else None
                            z.extractall(tmp_dir, pwd=pwd_bytes)
                        await safe_edit(main_msg, "🔄 Processing: Unzip...\n✅ Done!")
                        context.user_data["extracted_dir"] = tmp_dir
                    else:
                        await safe_edit(main_msg, "🔄 Processing: Unzip...\n❌ Not a zip file.")
                except Exception as e:
                    await safe_edit(main_msg, f"🔄 Processing: Unzip...\n❌ Failed: {str(e)}")

            elif action == "get_cc":
                await safe_edit(main_msg, "🔄 Processing: Get CC...\n✅ Extracting CreditCards...")

                base_name = os.path.splitext(os.path.basename(dest_path))[0]
                timestamped_folder = get_timestamped_folder(base_name)
                result_path = os.path.join(timestamped_folder, f"{base_name}_CreditCards_merged.txt")

                loop = asyncio.get_running_loop()
                try:
                    merged_files = await loop.run_in_executor(
                        None, merge_creditcards_from_archive, dest_path, result_path, pwd
                    )
                    print(f"[DEBUG] CC: merged_files={merged_files}, path={result_path}")
                except Exception as e:
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Extraction error: {str(e)}")
                    print(f"[DEBUG] CC extraction error: {e}")
                    continue
                
                if merged_files == 0:
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ No CreditCards found (count=0).")
                    print(f"[DEBUG] CC: merged_files is 0")
                    continue
                
                if not os.path.exists(result_path):
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Result file not created.")
                    print(f"[DEBUG] CC: file not created at {result_path}")
                    continue

                await safe_edit(main_msg, f"🔄 Processing: Get CC...\n✅ Got CC ({merged_files} line(s)), sending...")

                try:
                    with open(result_path, "rb") as f:
                        file_content = f.read()
                        print(f"[DEBUG] CC: file size={len(file_content)} bytes")
                        if file_content:
                            await update.message.reply_document(
                                document=InputFile(file_content, filename=os.path.basename(result_path)),
                                caption=f"✅ CreditCards ({merged_files} line(s))",
                            )
                            await safe_edit(main_msg, f"🔄 Processing: Get CC...\n✅ CC file sent!")
                        else:
                            await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ File is empty.")
                            print(f"[DEBUG] CC: file is empty")
                except Exception as e:
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Send error: {str(e)}")
                    print(f"[DEBUG] CC send error: {e}")

            elif action == "get_combo":
                if not keyword:
                    await safe_edit(main_msg, "⏭️ Skipping Get Combo (no keyword).")
                    continue
                
                await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n✅ Extracting...")

                base_name = os.path.splitext(os.path.basename(dest_path))[0]
                timestamped_folder = get_timestamped_folder(base_name)
                result_path = os.path.join(timestamped_folder, f"{base_name}_combo_{keyword}.txt")

                loop = asyncio.get_running_loop()
                try:
                    combo_count = await loop.run_in_executor(
                        None, extract_combos_from_archive, dest_path, keyword, result_path, pwd
                    )
                    print(f"[DEBUG] Combo: combo_count={combo_count}, path={result_path}")
                except Exception as e:
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Extraction error: {str(e)}")
                    print(f"[DEBUG] Combo extraction error: {e}")
                    continue

                if combo_count == 0:
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ No combos found (count=0).")
                    print(f"[DEBUG] Combo: combo_count is 0")
                    continue
                
                if not os.path.exists(result_path):
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Result file not created.")
                    print(f"[DEBUG] Combo: file not created at {result_path}")
                    continue

                await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n✅ Got Combo ({combo_count} found), sending...")
                try:
                    with open(result_path, "rb") as f:
                        file_content = f.read()
                        print(f"[DEBUG] Combo: file size={len(file_content)} bytes")
                        if file_content:
                            await update.message.reply_document(
                                document=InputFile(file_content, filename=os.path.basename(result_path)),
                                caption=f"✅ Combos for '{keyword}' ({combo_count} found)",
                            )
                            await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n✅ Combo file sent!")
                        else:
                            await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ File is empty.")
                            print(f"[DEBUG] Combo: file is empty")
                except Exception as e:
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Send error: {str(e)}")
                    print(f"[DEBUG] Combo send error: {e}")

            elif action == "cookies":
                await safe_edit(main_msg, "⏭️ Get Cookies (placeholder - coming soon).")

            elif action == "ulp":
                await safe_edit(main_msg, "⏭️ Get ULP (placeholder - coming soon).")
        
        except Exception as e:
            await safe_edit(main_msg, f"❌ Action error: {str(e)}")

    if main_msg:
        await safe_edit(main_msg, "✅ All selected actions completed.")
    context.user_data.clear()
