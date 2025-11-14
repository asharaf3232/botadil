module.exports = {
  apps : [{
    // OKX Bot Configuration
    name   : "okx-bot",
    script : "/root/bots/okx/index.js",
    args: "-r dotenv/config",
    cwd: "/root/bots/okx/"
  }, {
    // MCXC Bot Configuration
    name   : "mcxc-bot",
    script : "/root/bots/mcxc/fomo_hunter_bot.py",
    interpreter: "/root/bots/mcxc/venv/bin/python",
    env: {
      "TELEGRAM_BOT_TOKEN": "8237437309:AAEpVLo0DTvwLGevEXX5bAaKLlrNwmjVpWg",
      "TELEGRAM_CHANNEL_ID": "-1003098624744",
      "TELEGRAM_CHAT_ID": "1245603051"
    }
  }, {
    // MEXC Telegram Bot Configuration
    name   : "mexc-telegram-bot",
    script : "/root/bots/mexc-telegram-bot/main.py",
    interpreter: "/root/bots/mexc-telegram-bot/venv/bin/python",
    env: {
      "MEXC_API_KEY": "mx0vglImsrOWSr2LSS",
      "MEXC_API_SECRET": "a9ade801c2c84a39b261919a2fa2b00f",
      "TELEGRAM_BOT_TOKEN": "8485753576:AAEJDH7rBz0xHjS-hgiKQJ2JOnbp9EhvlI0",
      "TELEGRAM_CHAT_ID": "1245603051",
      "ALPHA_VANTAGE_API_KEY": "WBG43U84GFRSV2KI",
      "BINANCE_API_KEY": "c6P15N1gJHZwdeqXXzcHJSr2e5sCxb80AG3QeIVdQqsOl0MpktH5cixUuDKwUxo9",
      "BINANCE_API_SECRET": "bzwrmA5fHBAI1YXakNRoFr5zq50VhfH42cP8oiLDWVJqoqHecKW6rKV37XwQrIit",
      "TELEGRAM_SIGNAL_CHANNEL_ID": "-1002960540896"
    }
  }, { // <--- هذه هي الفاصلة التي كانت ناقصة
    // BinanceTraderBot (botadil) Configuration
    name   : "BinanceTraderBot",
    script : "/root/bots/botadil/binance_trader.py",
    interpreter: "/root/bots/botadil/venv/bin/python",
    cwd: "/root/bots/botadil/",
    env: {
       "TELEGRAM_BOT_TOKEN": "7649770299:AAHaBvKjCkrN6Up-D2jpuPX9t4aQnHrVz6g",
      "TELEGRAM_CHAT_ID": "1245603051",
      "TELEGRAM_SIGNAL_CHANNEL_ID": "-1002774814087",
      "ALPHA_VANTAGE_API_KEY": "WBG43U84GFRSV2KI",
      "BINANCE_API_KEY": "c6P15N1gJHZwdeqXXzcHJSr2e5sCxb80AG3QeIVdQqsOl0MpktH5cixUuDKwUxo9",
      "BINANCE_API_SECRET": "bzwrmA5fHBAI1YXakNRoFr5zq50VhfH42cP8oiLDWVJqoqHecKW6rKV37XwQrIit",
      "GATE_API_KEY": "8d3afd39a04abb779cd4851cfe15f414",
      "GATE_API_SECRET": "e172e30baaaf0c408d23fe0e06fbcf336c9a4338f44b78825a077de840b4ee03",
      "OKX_API_KEY": "39b0d754-221f-42d4-94ce-5e0fb68088e6",
      "OKX_API_SECRET": "868109A41992DB77CECF1B661ADF39AE",
      "OKX_API_PASSPHRASE": "2@Ashraf",
      "BYBIT_API_KEY": "ao8mG4SppJdk93jhQN",
      "BYBIT_API_SECRET": "d9GpJFBH7uoJeI2vAXXmaTbRIxBzTYTbwOCE"
    }
  }]
}
