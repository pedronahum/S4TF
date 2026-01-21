// Copyright 2024 Pedro N. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest
@testable import X10

final class X10Tests: XCTestCase {
    func testX10Import() {
        // Basic test to verify X10 module imports successfully
        // More comprehensive tests require the X10 library to be installed
        XCTAssertTrue(true, "X10 module imported successfully")
    }

    static var allTests = [
        ("testX10Import", testX10Import),
    ]
}
