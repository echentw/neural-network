#include <ostream>

namespace Color {
  enum Code {
    FG_RED      = 31,
    FG_GREEN    = 32,
    FG_BLUE     = 34,
    FG_DEFAULT  = 39,
    BG_RED      = 41,
    BG_GREEN    = 42,
    BG_BLUE     = 44,
    BG_DEFAULT  = 49
  };
  class Modifier {
    Code code;
    bool bold = false;
   public:
    Modifier(Code pCode) : code(pCode) {}
    Modifier(Code pCode, bool pBold) : code(pCode) {
      bold = pBold;
    }
    friend std::ostream&
    operator<<(std::ostream& os, const Modifier& mod) {
      return os << "\033[" << mod.bold << ";" << mod.code << "m";
    }
  };
}

void printPass(std::string test_name) {
  Color::Modifier g(Color::FG_GREEN);
  Color::Modifier b(Color::FG_BLUE);
  
  Color::Modifier def(Color::FG_DEFAULT);
  std::cout << b << test_name << g << " passes!" << def << std::endl;
}

