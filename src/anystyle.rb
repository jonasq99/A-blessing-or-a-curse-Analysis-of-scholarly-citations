if ARGV.length != 1
  puts "Usage: ruby example.rb 'Your Text Here'"
  exit
end

require "anystyle"

# Accessing the first argument
input_text = ARGV[0]


# parse the input text with anystyle and use a different output format



puts AnyStyle.parse input_text, format: :bibtex