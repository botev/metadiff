//
// Created by alex on 19/03/15.
//

#ifndef METADIFF_METADIFF_H
#define METADIFF_METADIFF_H

#include <memory>
#include <iostream>
#include <iomanip>
#include <exception>
#include <fstream>
#include <dlfcn.h>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include <unordered_map>

using std::vector;
using std::stack;
using std::string;
using std::unordered_set;
using std::unordered_map;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::map;
using std::set;
using std::cout;
using std::endl;

#include "os.h"
#include "logging.h"
#include "symbolic.h"
#include "defs.h"
#include "shared.h"
#include "core.h"
#include "exceptions.h"
#include "operators.h"
#include "optimize.h"
#include "core_impl.h"
#include "visual.h"
#include "backends.h"
#include "api.h"


#endif //METADIFF_METADIFF_H